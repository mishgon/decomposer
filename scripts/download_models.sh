#!/usr/bin/env bash
set -euo pipefail

uv tool run --from huggingface-hub hf download Qwen/Qwen3.5-4B
