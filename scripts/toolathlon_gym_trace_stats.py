import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-26B-A4B-it")


root = Path("artifacts") / "gyms" / "toolathlon_gym"


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
    } for message in messages]
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

percentages_passed = []

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

        native_result = ev["native_result"]
        if not (native_result is None) and "total_passed" in native_result and "total_checks" in native_result:
            percentage_passed = native_result["total_passed"] / native_result["total_checks"]
        percentages_passed.append(percentage_passed)

        if passed:
            successful_messages_length.append(l)
            successful_total_tokens.append(t)
            successful_delegations.append(d)
        else:
            wrong_messages_length.append(l)
            wrong_total_tokens.append(t)
            wrong_delegations.append(d)


print(f"Average messages length: {sum(messages_lengths) / len(messages_lengths)}")
if len(successful_messages_length) > 0:
    print(f"Passed messages length: {sum(successful_messages_length) / len(successful_messages_length)}")
if len(wrong_messages_length) > 0:
    print(f"Wrong messages length: {sum(wrong_messages_length) / len(wrong_messages_length)}")
print()
print(f"Average total tokens: {sum(total_tokens) / len(total_tokens)}")
if len(successful_total_tokens) > 0:
    print(f"Passed total tokens: {sum(successful_total_tokens) / len(successful_total_tokens)}")
if len(wrong_total_tokens) > 0:
    print(f"Wrong total tokens: {sum(wrong_total_tokens) / len(wrong_total_tokens)}")
print()
print(f"Average delegations: {sum(delegations) / len(delegations)}")
if len(successful_delegations) > 0:
    print(f"Passed delegations: {sum(successful_delegations) / len(successful_delegations)}")
if len(wrong_delegations) > 0:
    print(f"Wrong delegations: {sum(wrong_delegations) / len(wrong_delegations)}")



plt.figure(figsize=(8, 5))
weights = np.ones(len(percentages_passed)) / len(percentages_passed) * 100
plt.hist(percentages_passed, bins=10, range=(0, 1), weights=weights, edgecolor='black', color='#3498db')

plt.title(f"Distribution Percentage Chart; total {len(percentages_passed)}")
plt.xlabel('Value Intervals')
plt.ylabel('Percentage (%)')
plt.xticks(np.arange(0, 1.1, 0.1)) # Forces X-axis ticks at every 0.1
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig('pass_distribution.png', dpi=300)
plt.show()

print()
print("saved pass_distribution.png")
