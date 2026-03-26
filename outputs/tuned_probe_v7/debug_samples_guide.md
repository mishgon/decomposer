# `debug_samples.json` Guide

`debug_samples.json` is a compact inspection artifact produced for each dataset run.

## Top-level keys

- `raw_samples`: longest raw episodes selected by raw call count
- `abstract_samples`: longest model-facing episodes selected by abstract trajectory length

## `raw_samples` fields

- `path`: source JSON file under `raw_traces/`
- `episode_id`: stable dataset episode identifier
- `entry_function`: top-level SymPy API that was run
- `input`: original serialized input expression
- `final_output`: final serialized SymPy result
- `num_calls`: number of raw traced calls
- `pattern_label`: coarse shape label such as `chain`, `loop`, `hierarchy`, or `mixed`
- `call_sequence`: flattened raw call list in execution order

## `call_sequence` fields

- `call_id`: unique call node id inside the episode
- `depth`: tree depth
- `func_id`: traced SymPy function identity
- `output`: serialized display form of the return value

## `abstract_samples` fields

- same schema as one line in `abstract_traces/train.jsonl`
- `gold_trajectory` alternates assistant tool calls and tool observations
- each pair represents the ideal model-facing step sequence for that episode
