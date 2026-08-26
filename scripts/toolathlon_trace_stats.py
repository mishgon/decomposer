import os
import json
from pathlib import Path

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")


root = Path("artifacts") / "gyms" / "toolathlon"


evals = root / "evals"
traces = root / "traces"


successful_messages_length = []
wrong_messages_length = []
messages_lengths = []

total_tokens = []
successful_total_tokens = []
wrong_total_tokens = []

delegations = []
successful_delegations = []
wrong_delegations = []


role_mapping = {
        "human": "user",
        "ai": "assistant",
        "tool": "tool",
}

def count_tokens(messages):
    ms = [{
        **message,
        "role": role_mapping[message["type"]],
        "content": message["data"]["content"],
    } for message in messages if message["type"] in role_mapping]
    formatted_chat = tokenizer.apply_chat_template(
        ms,
        tokenize=False,
        add_generation_prompt=False,
    )
    tokens = tokenizer.encode(formatted_chat)
    t = len(tokens)
    return t


def count_delegations(messages):
    d = 0
    for message in messages:
        if message["type"] == "ai" and "tool_calls" in message["data"]:
            for tk in message["data"]["tool_calls"]:
                if tk["name"] == "spawn_subagent":
                    d += 1

    return d

all_tasks = os.listdir(traces)
for task in all_tasks:
    runs = os.listdir(traces / task)
    for run in runs:
        trace_filepath = traces / task / run / "trace.json"
        if not trace_filepath.is_file():
            continue
        with open(trace_filepath, "r") as f:
            trace = json.load(f)

        l = len(trace["messages"])
        t = count_tokens(trace["messages"])
        d = count_delegations(trace["messages"])

        messages_lengths.append(l)
        total_tokens.append(t)
        delegations.append(d)

        eval_filepath = evals / task / run / "result.json"
        if not eval_filepath.is_file():
            continue
        with open(eval_filepath, "r") as f:
            ev = json.load(f)
        passed = ev["pass"]
        if passed:
            successful_messages_length.append(l)
            successful_total_tokens.append(t)
            successful_delegations.append(d)
        else:
            wrong_messages_length.append(l)
            wrong_total_tokens.append(t)
            wrong_delegations.append(d)


print(f"Average messages length: {sum(messages_lengths) / len(messages_lengths)}")
print(f"Passed messages length: {sum(successful_messages_length) / len(successful_messages_length)}")
print(f"Wrong messages length: {sum(wrong_messages_length) / len(wrong_messages_length)}")
print()
print(f"Average total tokens: {sum(total_tokens) / len(total_tokens)}")
print(f"Passed total tokens: {sum(successful_total_tokens) / len(successful_total_tokens)}")
print(f"Wrong total tokens: {sum(wrong_total_tokens) / len(wrong_total_tokens)}")
print()
print(f"Average delegations: {sum(delegations) / len(delegations)}")
print(f"Passed delegations: {sum(successful_delegations) / len(successful_delegations)}")
print(f"Wrong delegations: {sum(wrong_delegations) / len(wrong_delegations)}")
