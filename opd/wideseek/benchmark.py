"""Replay fixed WideSeek tokens through the actor; compare speed, memory and gradients."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


VARIANTS = ('baseline', 'kernels', 'fused', 'fast')


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def worker(args):
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
    from transformers import Qwen3_5ForConditionalGeneration
    from peft import LoraConfig, get_peft_model
    from omegaconf import OmegaConf
    from verl.models.transformers.monkey_patch import apply_monkey_patch
    from verl.utils.fsdp_utils import get_fsdp_wrap_policy
    from verl.utils.torch_functional import logprobs_from_logits

    if args.variant in ('kernels', 'fused', 'fast'):
        from opd.wideseek.fast_runtime import check
        check()
    torch.manual_seed(42)
    torch.cuda.set_device(0)
    label = f'{args.variant}-{args.length}' + (f'-b{args.batch_size}' if args.batch_size != 1 else '')
    dist.init_process_group('nccl', init_method=f'file://{args.root / (label + ".dist")}', rank=0, world_size=1)
    try:
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            args.model, dtype=torch.float32, attn_implementation='sdpa')
        model.config.use_cache = False
        if args.variant == 'reference':
            from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

            def fp32_recurrence(*args, **kwargs):
                with torch.autocast('cuda', enabled=False):
                    return torch_chunk_gated_delta_rule(*args, **kwargs)

            count = 0
            for module in model.modules():
                if hasattr(module, 'chunk_gated_delta_rule'):
                    module.chunk_gated_delta_rule = fp32_recurrence
                    count += 1
            assert count == 24, f'Expected 24 linear-attention layers, found {count}'
        apply_monkey_patch(model, use_remove_padding=False,
                          use_fused_kernels=args.variant in ('head', 'fused', 'fast'), fused_kernels_backend='torch')
        model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, target_modules='all-linear',
                                               lora_dropout=0.0, task_type='CAUSAL_LM'))
        if args.variant != 'fast':
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            model.enable_input_require_grads()
        model = FSDP(model, device_id=0, use_orig_params=False,
                     auto_wrap_policy=get_fsdp_wrap_policy(model, OmegaConf.create({'min_num_params': 0}), is_lora=True),
                     mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,
                                                    reduce_dtype=torch.float32, buffer_dtype=torch.float32))
        model.train()
        trajectory = json.loads((args.root / 'trajectory.json').read_text())
        tokens = trajectory['prompt_ids'] + trajectory['response_ids']
        mask = [0] * len(trajectory['prompt_ids']) + trajectory['response_mask']
        # Repeat the real response only for the explicit maximum-length capacity case.
        while len(tokens) < args.length:
            tokens += trajectory['response_ids']
            mask += trajectory['response_mask']
        ids = torch.tensor([tokens[:args.length]], device='cuda').repeat(args.batch_size, 1)
        response_mask = torch.tensor([mask[:args.length]], device='cuda').roll(-1, dims=-1).repeat(args.batch_size, 1)
        response_mask[:, -1] = 0
        labels = ids.roll(-1, dims=-1)
        weights = torch.cos(torch.arange(args.length, device='cuda').float())[None] * response_mask
        rows = []
        for step in range(3):
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                kwargs = {'temperature': 0.7} if args.variant in ('head', 'fused', 'fast') else {}
                output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, **kwargs)
                log_probs = (output.log_probs if args.variant in ('head', 'fused', 'fast') else
                             logprobs_from_logits(output.logits / 0.7, labels))
                # Fixed signed weights isolate numerical equivalence from teacher/sampling noise.
                loss = (log_probs * weights).sum() / response_mask.sum()
            loss.backward()
            torch.cuda.synchronize()
            rows.append({'step': step, 'warmup': step == 0, 'seconds': time.perf_counter() - started,
                         'loss': float(loss.detach()),
                         'peak_gib': torch.cuda.max_memory_allocated() / 2**30})
            save(args.root / f'{label}.json', {'variant': args.variant, 'length': args.length,
                                              'batch_size': args.batch_size, 'steps': rows})
            print(json.dumps(rows[-1]), flush=True)
            if step == 2:
                gradients = {name: p.grad.detach().float().cpu() for name, p in model.named_parameters()
                             if p.requires_grad and p.grad is not None}
                torch.save({'gradients': gradients, 'log_probs': log_probs.detach().float().cpu()},
                           args.root / f'{label}.pt')
            del output, log_probs, loss
    finally:
        dist.destroy_process_group()


def run(args):
    import fcntl
    from opd.wideseek.fast_runtime import environment
    args.root.mkdir(parents=True, exist_ok=False)
    with (args.root.parent / f'.benchmark-gpu-{args.gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        memory = int(subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
            '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
        if memory > 1024:
            raise RuntimeError(f'GPU {args.gpu} is occupied ({memory} MiB)')
        candidates = [(p, json.loads(p.read_text())) for p in args.episodes.glob('*/trajectory.json')]
        path, trajectory = max(((p, d) for p, d in candidates if d['data_source'] == 'wideseek/train'),
                               key=lambda pair: len(pair[1]['prompt_ids']) + len(pair[1]['response_ids']))
        (args.root / 'trajectory.json').write_text(json.dumps(trajectory))
        state = {'status': 'running', 'started_at': time.time(), 'gpu': args.gpu,
                 'source': str(path), 'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                 'active': 'preflight', 'completed': 0, 'total': 8, 'results': []}
        save(args.root / 'status.json', state)
        try:
            for length in (8192, 16384):
                for variant in VARIANTS:
                    state['active'] = f'{variant}-{length}'
                    save(args.root / 'status.json', state)
                    env = dict(os.environ) if variant == 'baseline' else environment(args.bundle)
                    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false')
                    command = [sys.executable, '-u', '-m', 'opd.wideseek.benchmark', '--variant', variant,
                               '--length', str(length), '--model', str(args.model), '--root', str(args.root)]
                    with (args.root / f'{variant}-{length}.log').open('w') as log:
                        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                    import torch
                    result = json.loads((args.root / f'{variant}-{length}.json').read_text())
                    result['median_seconds'] = statistics.median(r['seconds'] for r in result['steps'][1:])
                    result['peak_gib'] = max(r['peak_gib'] for r in result['steps'])
                    if variant != 'baseline':
                        reference = torch.load(args.root / f'baseline-{length}.pt', map_location='cpu', weights_only=True)
                        candidate = torch.load(args.root / f'{variant}-{length}.pt', map_location='cpu', weights_only=True)
                        assert reference['gradients'].keys() == candidate['gradients'].keys()
                        dot = norm_a = norm_b = error = 0.0
                        for name, a in reference['gradients'].items():
                            b = candidate['gradients'][name]
                            dot += (a.double() * b).sum().item()
                            norm_a += a.double().square().sum().item()
                            norm_b += b.double().square().sum().item()
                            error += (a.double() - b).square().sum().item()
                        result['gradient_cosine'] = dot / (norm_a * norm_b)**0.5
                        result['relative_gradient_error'] = (error / norm_a)**0.5
                        result['max_logprob_error'] = float((reference['log_probs'] - candidate['log_probs']).abs().max())
                        result['numerical_check'] = result['gradient_cosine'] > .999 and result['relative_gradient_error'] < .05
                        del reference, candidate
                    state['results'].append(result)
                    state['completed'] += 1
                    save(args.root / 'status.json', state)
            state['status'] = 'completed'
        except BaseException as exc:
            state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            state['finished_at'] = time.time()
            save(args.root / 'status.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--episodes', type=Path)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--variant', choices=(*VARIANTS, 'reference', 'head'))
    parser.add_argument('--length', type=int)
    parser.add_argument('--batch-size', type=int, default=1)
    args = parser.parse_args()
    worker(args) if args.variant else run(args)
