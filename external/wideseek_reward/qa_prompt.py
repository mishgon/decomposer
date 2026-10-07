# Copyright 2025 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (see LICENSE in this directory).
# Copied from utils/prompt.py at RLinf 64875d346d5cafb06c1112f563b20d2c6360bfae.

BOXED_FORMAT_EN = "If you determine that no further external knowledge is required, you have to wrap your final answer in \\boxed{}."

LLM_JUDGE_PROMPT = """Question: {question}

Labeled Answer: {correct_answer}

Predicted Answer: {response}

Did the model give an answer **equivalent** to the labeled answer?

Please respond with "Correct" if they are equivalent, or "Incorrect" if they are not equivalent. Do not include any other text."""
