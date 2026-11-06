"""Bounded vLLM weight-load benchmark; excludes actor export and NCCL transport."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time

from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension


class SyncBenchmarkWorker(vLLMColocateWorkerExtension):
    def bench_prepare(self, model_path, checkpoint):
        return prepare(self, model_path, checkpoint)

    def bench_reload(self, kind):
        return reload(self, kind)


def prepare(worker, model_path, checkpoint):
    import torch
    from safetensors.torch import load_file
    from verl.utils.vllm import VLLMHijack
    VLLMHijack.hijack()
    adapters = torch.load(checkpoint, map_location='cpu', weights_only=True)
    adapters = {k.replace('.default.', '.'): v for k, v in adapters.items()}
    base = {}
    for path in sorted(Path(model_path).glob('*.safetensors')):
        base.update(load_file(str(path)))
    assert base and adapters
    merged = {k: v.to(torch.bfloat16) for k, v in base.items()}
    for name, a in adapters.items():
        if '.lora_A.' not in name:
            continue
        key = name.removeprefix('base_model.model.').replace('.lora_A.', '.')
        b = adapters[name.replace('.lora_A.', '.lora_B.')]
        assert key in base, key
        merged[key] = (base[key].float() + 2.0 * (b.float() @ a.float())).to(torch.bfloat16)
    # Keep host payloads resident; disk reading and LoRA merge are outside load timings.
    worker.bench_payloads = {'base': {k: v.to(torch.bfloat16) for k, v in base.items()},
                             'merged': merged, 'adapter': adapters,
                             'zero': {k: torch.zeros_like(v) for k, v in adapters.items()}}
    worker.bench_peft = {'r': 32, 'lora_alpha': 64, 'peft_type': 'LORA',
                         'task_type': 'CAUSAL_LM', 'target_modules': sorted({k.split('.')[-3] for k in adapters}),
                         'bias': 'none'}
    return {kind: sum(t.numel() * t.element_size() for t in tensors.values())
            for kind, tensors in worker.bench_payloads.items()}


def reload(worker, kind):
    import torch
    from verl.workers.rollout.vllm_rollout.utils import VLLM_LORA_INT_ID
    torch.cuda.synchronize()
    start = time.perf_counter()
    worker.remove_lora(VLLM_LORA_INT_ID)
    tensors = [(k, v.to('cuda')) for k, v in worker.bench_payloads[kind].items()]
    adapter = kind in ('adapter', 'zero')
    worker._update_weights(tensors, worker.bench_peft if adapter else None, adapter)
    torch.cuda.synchronize()
    return time.perf_counter() - start


def main(args):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from opd.wideseek.benchmark import save
    assert list(args.model.glob('*.safetensors')), 'No model shards found'
    assert args.checkpoint.is_file() and args.trajectory.is_file()
    args.root.mkdir(parents=True, exist_ok=False)
    state = dict(status='running', started_at=time.time(), gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                 completed=0, total=5, active='loading vLLM', results=[],
                 notes=['Receiver-load benchmark only: host-to-GPU copy + vLLM reload; excludes actor export and NCCL.'])
    def update():
        save(args.root / 'status.json', state)
    update()
    llm = None
    try:
        llm = LLM(model=str(args.model), dtype='bfloat16', enforce_eager=True,
                  max_model_len=1024, max_num_seqs=1, gpu_memory_utilization=0.3,
                  enable_lora=True, max_lora_rank=32, enable_prefix_caching=False,
                  language_model_only=True,
                  worker_extension_cls='opd.wideseek.benchmark_sync.SyncBenchmarkWorker',
                  additional_config={'gdn_prefill_backend': 'triton'})
        payloads = llm.collective_rpc('bench_prepare', args=(str(args.model), str(args.checkpoint)))[0]
        trajectory = json.loads(args.trajectory.read_text())
        ids = (trajectory['prompt_ids'] + trajectory['response_ids'])[:256]
        samples = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1)
        values, timings = {}, {}
        for kind in ('base', 'merged', 'zero', 'adapter', 'adapter_reload'):
            state['active'] = kind
            update()
            actual = 'adapter' if kind == 'adapter_reload' else kind
            if actual in ('zero', 'adapter'):
                llm.collective_rpc('bench_reload', args=('base',))
            times = [llm.collective_rpc('bench_reload', args=(actual,))[0] for _ in range(4)]
            request = LoRARequest('123', 123, 'simon_lora_path') if actual in ('zero', 'adapter') else None
            result = llm.generate([{'prompt_token_ids': ids}], samples, lora_request=request, use_tqdm=False)[0]
            values[kind] = [p[token].logprob for token, p in zip(ids[1:], result.prompt_logprobs[1:])]
            timings[kind] = {'seconds': times, 'median_seconds': statistics.median(times[1:])}
            state['completed'] += 1
            state['notes'].append(f'{kind}: {timings[kind]["median_seconds"]:.3f}s median receiver load')
            update()
        def compare(a, b):
            errors = [abs(x-y) for x,y in zip(values[a],values[b])]
            return {'mean_abs_logprob_error': statistics.mean(errors), 'max_abs_logprob_error': max(errors)}
        comparisons = {f'{a}_vs_{b}': compare(a,b) for a,b in
                       [('adapter','merged'),('zero','base'),('adapter_reload','adapter'),('adapter','base')]}
        result = {'payload_bytes': payloads, 'timings': timings, 'comparisons': comparisons,
                  'logprobs': values, 'checkpoint': str(args.checkpoint),
                  'scope': 'Receiver only; no actor export, NCCL, optimizer or teacher calls.'}
        save(args.root / 'result.json', result)
        print(json.dumps(result, indent=2), flush=True)
        state['status'] = 'completed'
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        state['finished_at'] = time.time()
        update()
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'model', 'checkpoint', 'trajectory'):
        parser.add_argument('--' + name, type=Path, required=True)
    main(parser.parse_args())
