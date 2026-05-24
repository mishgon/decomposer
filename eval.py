"""Evaluate Decomposer and direct baselines on MuSiQue dev."""

import argparse
import os
import re
import string
import unicodedata
from collections import Counter
from collections.abc import Callable
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

from prompts import (
    build_context_qa_messages,
    build_decomposer_messages,
    build_func_spec,
)
from prototype import (
    EXEC_BUILTINS,
    extract_answer,
    extract_func_body,
    format_context,
    is_unanswerable_answer,
    load_examples,
    render_func_src,
    validate_func_src,
)
from sampling import build_sampling_params as _build_sampling_params
from utils import write_jsonl


MUSIQUE_DEV_PATH = Path("data/musique_ans_v1.0_dev.jsonl")
DECOMPOSER_MODEL = "Qwen/Qwen3.6-27B"
DECOMPOSER_ENABLE_THINKING = True
QA_MODELS = (
    "Qwen/Qwen3.6-27B",
)
QA_ENABLE_THINKING = True
MAX_PRINTED_EXAMPLES = 20


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str | None, ground_truth: str) -> float:
    if prediction is None:
        return 0.0
    return float(normalize_text(prediction) == normalize_text(ground_truth))


def filter_examples_by_num_hops(
    examples: list[dict],
    num_hops: int | None,
) -> list[dict]:
    if num_hops is None:
        return examples
    return [
        example
        for example in examples
        if len(example["question_decomposition"]) == num_hops
    ]


def build_chat_prompt(llm: LLM, messages: list[dict], enable_thinking: bool) -> str:
    tokenizer = llm.get_tokenizer()
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def build_vllm_sampling_params(
    model: str,
    enable_thinking: bool,
    task: str,
) -> SamplingParams:
    return SamplingParams(
        **_build_sampling_params(model, enable_thinking, task=task),
        seed=0,
    )


def load_llm(model: str) -> LLM:
    return LLM(
        model=model,
        dtype="auto",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.5,
        max_model_len=16384,
        max_num_seqs=32,
        additional_config={"gdn_prefill_backend": "triton"},
    )


def release_llm(llm: LLM) -> None:
    del llm
    cleanup_dist_env_and_memory()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def format_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def generate_decompositions(
    examples: list[dict],
    model: str,
    enable_thinking: bool,
    output_path: Path,
) -> list[dict]:
    print(
        "\nGenerating decompositions: "
        f"{model}, enable_thinking={enable_thinking}"
    )
    llm = load_llm(model)
    try:
        func_specs = [
            build_func_spec(example["question"])
            for example in examples
        ]
        prompts = [
            build_chat_prompt(
                llm,
                build_decomposer_messages(func_spec),
                enable_thinking=enable_thinking,
            )
            for func_spec in func_specs
        ]
        outputs = llm.generate(
            prompts,
            build_vllm_sampling_params(model, enable_thinking, task="code"),
        )
        decompositions = []
        for example, func_spec, output in zip(examples, func_specs, outputs):
            raw_output = output.outputs[0].text
            func_body = extract_func_body(raw_output)
            func_src = None
            generation_error = None
            try:
                func_src = render_func_src(func_spec, func_body)
                validate_func_src(func_src)
            except Exception as exc:
                generation_error = format_error(exc)
            decompositions.append(
                {
                    "id": example["id"],
                    "decomposer_model": model,
                    "decomposer_enable_thinking": enable_thinking,
                    "func_spec": func_spec,
                    "raw_output": raw_output,
                    "func_body": func_body,
                    "func_src": func_src,
                    "generation_error": generation_error,
                }
            )
    finally:
        release_llm(llm)

    write_jsonl(output_path, decompositions)
    num_errors = sum(
        decomposition["generation_error"] is not None
        for decomposition in decompositions
    )
    print(f"Saved decompositions to {output_path}")
    print(f"Generation errors: {num_errors}/{len(decompositions)}")
    return decompositions


def exec_func(
    func_src: str,
    context: str,
    llm: LLM,
    model: str,
    enable_thinking: bool,
    on_context_qa_answer: Callable[[], None] | None = None,
) -> tuple[str | None, list[dict], str | None]:
    validate_func_src(func_src)
    namespace = {"__builtins__": EXEC_BUILTINS.copy()}
    exec(func_src, namespace)
    tool_calling_logs = []

    def context_qa_model(question: str, context: str) -> str:
        prompt = build_chat_prompt(
            llm,
            build_context_qa_messages(question, context),
            enable_thinking=enable_thinking,
        )
        output = llm.generate(
            [prompt],
            build_vllm_sampling_params(model, enable_thinking, task="general"),
            use_tqdm=False,
        )[0]
        raw_output = output.outputs[0].text
        answer = extract_answer(raw_output)
        tool_calling_logs.append(
            {
                "tool": "context_qa_model",
                "question": question,
                "raw_output": raw_output,
                "answer": answer,
            }
        )
        if on_context_qa_answer is not None:
            on_context_qa_answer()
        if not answer:
            raise ValueError("Context QA output did not contain an answer.")
        if is_unanswerable_answer(answer):
            raise ValueError("Context QA question is unanswerable.")
        return answer

    try:
        prediction = namespace["main"](context, context_qa_model)
    except Exception as exc:
        return None, tool_calling_logs, format_error(exc)
    if prediction is not None:
        prediction = str(prediction)
    return prediction, tool_calling_logs, None


def evaluate_decompositions(
    examples: list[dict],
    decompositions: list[dict],
    baseline_result: dict,
    llm: LLM,
    model: str,
    enable_thinking: bool,
) -> dict:
    if len(examples) != len(decompositions):
        raise ValueError("Number of examples and decompositions must match.")

    baseline_by_id = {
        example_result["id"]: example_result
        for example_result in baseline_result["examples"]
    }
    results = []

    with tqdm(
        desc=f"Context QA: {model}",
        unit="question",
        bar_format="{desc}: {n_fmt} questions [{elapsed}, {rate_fmt}{postfix}]",
        dynamic_ncols=True,
    ) as progress:
        for example, decomposition in zip(examples, decompositions):
            if example["id"] != decomposition["id"]:
                raise ValueError(
                    "Example and decomposition ids must match: "
                    f"{example['id']!r} != {decomposition['id']!r}"
                )
            if example["id"] not in baseline_by_id:
                raise ValueError(f"Missing baseline result for {example['id']!r}.")

            prediction = None
            tool_calling_logs = []
            execution_error = None
            generation_error = decomposition["generation_error"]
            func_src = decomposition["func_src"]

            if generation_error is None:
                try:
                    prediction, tool_calling_logs, execution_error = exec_func(
                        func_src,
                        format_context(example),
                        llm=llm,
                        model=model,
                        enable_thinking=enable_thinking,
                        on_context_qa_answer=progress.update,
                    )
                except Exception as exc:
                    execution_error = format_error(exc)

            covered = generation_error is None and execution_error is None
            decomposition_score = (
                exact_match(prediction, example["answer"])
                if covered
                else None
            )
            baseline_prediction = baseline_by_id[example["id"]]["prediction"]
            hybrid_prediction = prediction if covered else baseline_prediction
            hybrid_score = exact_match(hybrid_prediction, example["answer"])
            results.append(
                {
                    "id": example["id"],
                    "prediction": prediction,
                    "covered": covered,
                    "decomposition_exact_match": decomposition_score,
                    "baseline_prediction": baseline_prediction,
                    "hybrid_prediction": hybrid_prediction,
                    "hybrid_exact_match": hybrid_score,
                    "execution_error": execution_error,
                    "tool_calling_logs": tool_calling_logs,
                }
            )

    covered_results = [result for result in results if result["covered"]]
    coverage = len(covered_results) / len(results)
    baseline_exact_match_score = baseline_result["exact_match"]
    baseline_covered_exact_match_score = (
        sum(baseline_by_id[result["id"]]["exact_match"] for result in covered_results)
        / len(covered_results)
        if covered_results
        else None
    )
    decomposition_exact_match_score = (
        sum(result["decomposition_exact_match"] for result in covered_results)
        / len(covered_results)
        if covered_results
        else None
    )
    hybrid_exact_match_score = (
        sum(result["hybrid_exact_match"] for result in results)
        / len(results)
    )
    generation_error_count = sum(
        decomposition["generation_error"] is not None
        for decomposition in decompositions
    )
    execution_error_counts = Counter(
        result["execution_error"]
        for result in results
        if result["execution_error"] is not None
    )
    execution_error_count = sum(execution_error_counts.values())

    print(f"\nDecomposer: QA model={model}, enable_thinking={enable_thinking}")
    print(f"Coverage: {coverage:.4f} ({len(covered_results)}/{len(results)})")
    print(f"Direct Baseline Exact Match: {baseline_exact_match_score:.4f}")
    if baseline_covered_exact_match_score is None:
        print("Direct Baseline Exact Match On Covered: n/a")
    else:
        print(
            "Direct Baseline Exact Match On Covered: "
            f"{baseline_covered_exact_match_score:.4f}"
        )
    if decomposition_exact_match_score is None:
        print("Decomposition Exact Match: n/a")
    else:
        print(f"Decomposition Exact Match: {decomposition_exact_match_score:.4f}")
    print(f"Hybrid Exact Match: {hybrid_exact_match_score:.4f}")
    print(f"Generation errors: {generation_error_count}/{len(results)}")
    print(f"Execution errors: {execution_error_count}/{len(results)}")
    if execution_error_counts:
        print("Execution error types:")
        for error, count in execution_error_counts.most_common():
            print(f"- {error}: {count}")
    if len(results) <= MAX_PRINTED_EXAMPLES:
        for example, decomposition, result in zip(examples, decompositions, results):
            decomposition_score = (
                "n/a"
                if result["decomposition_exact_match"] is None
                else f"{result['decomposition_exact_match']:.0f}"
            )
            print(
                f"- {result['id']}: decomp_EM={decomposition_score} "
                f"covered={result['covered']} "
                f"hybrid_EM={result['hybrid_exact_match']:.0f} "
                f"pred={result['prediction']!r} gold={example['answer']!r} "
                f"gen_error={decomposition['generation_error']!r} "
                f"exec_error={result['execution_error']!r}"
            )
    return {
        "method": "decomposer",
        "decomposer_model": (
            decompositions[0]["decomposer_model"]
            if decompositions
            else None
        ),
        "decomposer_enable_thinking": (
            decompositions[0]["decomposer_enable_thinking"]
            if decompositions
            else None
        ),
        "qa_model": model,
        "qa_enable_thinking": enable_thinking,
        "num_examples": len(results),
        "baseline_exact_match": baseline_exact_match_score,
        "baseline_covered_exact_match": baseline_covered_exact_match_score,
        "decomposition_exact_match": decomposition_exact_match_score,
        "coverage": coverage,
        "num_covered": len(covered_results),
        "hybrid_exact_match": hybrid_exact_match_score,
        "generation_error_count": generation_error_count,
        "execution_error_count": execution_error_count,
        "execution_error_counts": dict(execution_error_counts),
        "examples": results,
    }


def evaluate_direct_baseline(
    examples: list[dict],
    llm: LLM,
    model: str,
    enable_thinking: bool,
) -> dict:
    prompts = [
        build_chat_prompt(
            llm,
            build_context_qa_messages(example["question"], format_context(example)),
            enable_thinking=enable_thinking,
        )
        for example in examples
    ]
    outputs = llm.generate(
        prompts,
        build_vllm_sampling_params(model, enable_thinking, task="general"),
    )
    predictions = [extract_answer(output.outputs[0].text) for output in outputs]
    predictions = [
        None if prediction and is_unanswerable_answer(prediction) else prediction
        for prediction in predictions
    ]
    scores = [
        exact_match(prediction, example["answer"])
        for prediction, example in zip(predictions, examples)
    ]

    print(f"\nDirect baseline: {model}, enable_thinking={enable_thinking}")
    exact_match_score = sum(scores) / len(scores)
    print(f"Exact Match: {exact_match_score:.4f} (all examples)")
    if len(examples) <= MAX_PRINTED_EXAMPLES:
        for example, prediction, score in zip(examples, predictions, scores):
            print(
                f"- {example['id']}: EM={score:.0f} "
                f"pred={prediction!r} gold={example['answer']!r}"
            )
    return {
        "method": "direct_baseline",
        "model": model,
        "enable_thinking": enable_thinking,
        "num_examples": len(examples),
        "exact_match": exact_match_score,
        "examples": [
            {
                "id": example["id"],
                "question": example["question"],
                "prediction": prediction,
                "answer": example["answer"],
                "exact_match": score,
            }
            for example, prediction, score in zip(examples, predictions, scores)
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=MUSIQUE_DEV_PATH)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--num-hops", type=int, choices=(2, 3, 4))
    parser.add_argument("--decomposer-model", default=DECOMPOSER_MODEL)
    parser.add_argument(
        "--decomposer-thinking",
        action=argparse.BooleanOptionalAction,
        default=DECOMPOSER_ENABLE_THINKING,
    )
    parser.add_argument(
        "--qa-thinking",
        action=argparse.BooleanOptionalAction,
        default=QA_ENABLE_THINKING,
    )
    parser.add_argument("--qa-models", nargs="+", default=list(QA_MODELS))
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_examples(args.data)
    examples = filter_examples_by_num_hops(examples, args.num_hops)
    if args.limit is not None:
        examples = examples[:args.limit]
    if not examples:
        raise ValueError("No examples selected; adjust --data, --limit, or --num-hops.")

    print(f"Dev examples: {len(examples)}")
    if args.num_hops is not None:
        print(f"Question hops: {args.num_hops}")
    print(
        "Decomposer model: "
        f"{args.decomposer_model} (enable_thinking={args.decomposer_thinking})"
    )
    print("QA / baseline models:")
    for model in args.qa_models:
        print(f"- {model}")
    print(f"QA enable_thinking: {args.qa_thinking}")

    first = examples[0]
    print("\nFirst dev example:")
    print(f"ID: {first['id']}")
    print(f"Question: {first['question']}")
    print(f"Answer: {first['answer']}")
    print(f"Context characters: {len(format_context(first))}")

    decompositions = generate_decompositions(
        examples,
        model=args.decomposer_model,
        enable_thinking=args.decomposer_thinking,
        output_path=args.output_dir / "decompositions.jsonl",
    )

    results = []
    for model in args.qa_models:
        llm = load_llm(model)
        try:
            baseline_result = evaluate_direct_baseline(
                examples,
                llm=llm,
                model=model,
                enable_thinking=args.qa_thinking,
            )
            results.append(baseline_result)
            results.append(
                evaluate_decompositions(
                    examples,
                    decompositions,
                    baseline_result=baseline_result,
                    llm=llm,
                    model=model,
                    enable_thinking=args.qa_thinking,
                )
            )
        finally:
            release_llm(llm)

    write_jsonl(args.output_dir / "results.jsonl", results)
    print(f"\nSaved results to {args.output_dir}/results.jsonl")


if __name__ == "__main__":
    main()
