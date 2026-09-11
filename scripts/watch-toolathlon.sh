#!/usr/bin/env bash

set -u

FIXED_RUN_ID="${1:-}"
ONCE=0
if [[ "$FIXED_RUN_ID" == "--once" ]]; then ONCE=1; FIXED_RUN_ID=""; fi
INTERVAL="${WATCH_INTERVAL:-15}"
REPO="${DECOMPOSER_REPO:-$HOME/decomposer-rl}"
ROOT="${TOOLATHLON_ROOT:-$REPO/artifacts/gyms/toolathlon_gym}"
PYTHON="${WATCH_PYTHON:-$REPO/.venv-rl/bin/python}"
DOCKER="${CONTAINER_ENGINE:-podman}"
cd "$REPO" || exit 1

while true; do
  if [[ -n "$FIXED_RUN_ID" ]]; then
    RUN_ID="$FIXED_RUN_ID"
    MANIFEST="$ROOT/$RUN_ID/manifest.json"
    [[ -f "$MANIFEST" ]] || MANIFEST="$ROOT/runs/$RUN_ID/manifest.json"
  elif [[ -f "$ROOT/manifest.json" ]]; then
    MANIFEST="$ROOT/manifest.json"
    RUN_ID="$(basename "$ROOT")"
  else
    MANIFEST="$(find "$ROOT" -maxdepth 3 -name manifest.json \
      -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk 'NR==1 {print $2}')"
    RUN_ID="$(basename "$(dirname "${MANIFEST:-unknown}")")"
  fi

  [[ "$ONCE" == 1 ]] || clear
  printf 'TOOLATHLON GYM  %s  (updated %s)\n' "$RUN_ID" "$(date --utc '+%H:%M:%S UTC')"
  printf '%s\n' '----------------------------------------------------------------------'

  if [[ -z "${MANIFEST:-}" || ! -f "$MANIFEST" ]]; then
    printf 'No run manifest found under %s/runs\n' "$ROOT"
    [[ "$ONCE" == 1 ]] && exit 1
    sleep "$INTERVAL"
    continue
  fi

  "$PYTHON" - "$MANIFEST" <<'PY'
import json
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


def parse_time(value):
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, timezone.utc)
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def duration_text(seconds):
    seconds = max(0, int(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{days}d {hours:02d}h {minutes:02d}m"
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m {seconds:02d}s"


manifest_path = Path(sys.argv[1])
manifest = json.loads(manifest_path.read_text())
episodes = manifest["episodes"]
new_format = isinstance(episodes, dict)
if new_format:
    episodes = [{**e, "task": key.split("/rep-")[0],
                 "score": e.get("passed"), "duration_seconds": e.get("elapsed_seconds"),
                 "status": "failed" if e["status"] == "infrastructure_error" else e["status"]}
                for key, e in episodes.items()]
    total = len(manifest["tasks"]) * manifest["config"]["repetitions"]
    episodes += [{"status": "pending"}] * (total - len(episodes))
counts = {
    status: sum(episode.get("status") == status for episode in episodes)
    for status in ("pending", "running", "completed", "failed")
}
terminal = [
    episode for episode in episodes
    if episode.get("status") in {"completed", "failed"}
]
scored = [episode for episode in episodes if isinstance(episode.get("score"), bool)]
passed = sum(episode.get("score") is True for episode in scored)
remaining = counts["pending"] + counts["running"]

now = datetime.now(timezone.utc)
started = parse_time(manifest.get("created_at", manifest.get("started_at")))
end = parse_time(manifest["finished_at"]) if manifest.get("finished_at") else now
elapsed = max((end - started).total_seconds(), 0.0)
config = manifest.get("config", {})
concurrency = config.get("concurrency")
if not isinstance(concurrency, int):
    concurrency = 1
    for invocation in reversed(manifest.get("invocations", [])):
        argv = invocation.get("argv") or []
        for index, argument in enumerate(argv):
            if argument == "--concurrency" and index + 1 < len(argv):
                concurrency = int(argv[index + 1])
                break
            if argument.startswith("--concurrency="):
                concurrency = int(argument.split("=", 1)[1])
                break
        else:
            continue
        break

print(f"Status:     {manifest['status']} (concurrency {concurrency})")
print(f"Elapsed:    {duration_text(elapsed)} since {started:%Y-%m-%d %H:%M UTC}")
print(
    f"Progress:   {len(terminal)}/{len(episodes)} "
    f"({100 * len(terminal) / len(episodes):.1f}%)"
)
print(
    f"Queue:      {counts['pending']} pending | {counts['running']} running | "
    f"{counts['completed']} completed | {counts['failed']} {'infrastructure errors' if new_format else 'failed'}"
)
if scored:
    print(f"Quality:    {passed}/{len(scored)} passed ({100 * passed / len(scored):.1f}%)")
else:
    print("Quality:    waiting for first score")

durations = [
    float(episode["duration_seconds"])
    for episode in terminal
    if isinstance(episode.get("duration_seconds"), (int, float))
    and episode["duration_seconds"] > 0
]
if durations:
    print(
        f"Episodes:   median {statistics.median(durations) / 60:.1f}m | "
        f"mean {statistics.mean(durations) / 60:.1f}m | {len(durations)} samples"
    )

if terminal and elapsed > 0 and remaining:
    episodes_per_hour = len(terminal) * 3600 / elapsed
    eta_seconds = remaining / episodes_per_hour * 3600
    finish = now + timedelta(seconds=eta_seconds)
    print(
        f"ETA (whole run, approximate): {duration_text(eta_seconds)} | finish {finish:%Y-%m-%d %H:%M UTC} | "
        f"{episodes_per_hour:.1f} episodes/hour observed"
    )
elif not remaining:
    print("ETA:        complete")
else:
    print("ETA:        waiting for first finished episode")

input_tokens = output_tokens = reasoning_tokens = model_responses = 0
for episode in episodes:
    artifact = episode.get("artifact_path")
    if not artifact:
        continue
    try:
        totals = json.loads((Path(artifact) / "usage.json").read_text()).get("totals", {})
    except (OSError, json.JSONDecodeError):
        continue
    input_tokens += int(totals.get("input_tokens") or 0)
    output_tokens += int(totals.get("output_tokens") or 0)
    reasoning_tokens += int(totals.get("reasoning_tokens") or 0)
    model_responses += int(totals.get("model_responses") or 0)
if model_responses:
 print(
    f"Tokens:     {input_tokens / 1e6:.2f}M in | {output_tokens / 1e6:.2f}M out | "
    f"{reasoning_tokens / 1e6:.2f}M reasoning | {model_responses} responses"
)
else:
 print("Tokens:     usage totals unavailable (not zero)")

running = [episode for episode in episodes if episode.get("status") == "running"]
if running:
    rows = []
    for episode in running:
        try:
            age = (now - parse_time(episode["started_at"])).total_seconds()
            age_text = f"{age / 60:5.1f}m"
        except (KeyError, AttributeError, ValueError):
            age_text = "   n/a"
        rows.append(f"  {age_text}  {episode['task'][:58]}")
    print(f"\nRunning now ({len(rows)}; age includes container setup):")
    print("\n".join(rows[:12]))
    if len(rows) > 12:
        print(f"  ... {len(rows) - 12} more running")
PY

  read -r _configured_gpu subagent_ports <<<"$("$PYTHON" - "$MANIFEST" <<'PY'
import json
import sys
from urllib.parse import urlparse
manifest = json.load(open(sys.argv[1]))
config = manifest.get("config", {})
ports = config.get("subagent_ports") or [config.get("subagent_port", "")]
teacher_port = urlparse(config.get("url", "")).port
if teacher_port and teacher_port not in ports:
    ports.append(teacher_port)
print(config.get("subagent_gpu", "0"), ",".join(map(str, ports)))
PY
)"
  gpu_ids="${WATCH_GPUS:-0,2}"
  evaluator_state="stopped"
  if pgrep -f "(gyms/toolathlon_gym/run.py|gyms.toolathlon_gym.evaluate_sft).*${RUN_ID}" >/dev/null 2>&1; then
    evaluator_state="running"
  fi
  container_count="$($DOCKER ps --format '{{.Names}}' 2>/dev/null | wc -l)"
  load_one="$(cut -d' ' -f1 /proc/loadavg)"
  memory="$(free -h | awk '/^Mem:/{print $3 "/" $2}')"
  printf '\nLocal infrastructure:\n'
  printf '  evaluator %s | %s containers | host load %s | RAM %s\n' \
    "$evaluator_state" "$container_count" "$load_one" "$memory"

  if [[ -n "$gpu_ids" ]]; then
    while IFS=',' read -ra ids; do
      for gpu_id in "${ids[@]}"; do
        gpu_line="$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
          --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || true)"
        if [[ -n "$gpu_line" ]]; then
          IFS=',' read -r gpu_index gpu_used gpu_total gpu_util <<<"$gpu_line"
          printf '  GPU%s:%s/%s MiB | utilization%s%%\n' \
            "$gpu_index" "$gpu_used" "$gpu_total" "$gpu_util"
        fi
      done
    done <<<"$gpu_ids"
  fi

  model_ports="$subagent_ports"
  for subagent_port in ${model_ports//,/ }; do
    [[ -n "$subagent_port" ]] || continue
    metrics="$(curl -fsS --max-time 3 "http://127.0.0.1:${subagent_port}/metrics" 2>/dev/null || true)"
    if [[ -n "$metrics" ]]; then
      read -r active waiting completed length errors <<<"$(awk '
        /vllm:num_requests_running\{/{a=$NF}
        /vllm:num_requests_waiting\{/{w=$NF}
        /vllm:request_success_total.*finished_reason="stop"/{s+=$NF}
        /vllm:request_success_total.*finished_reason="length"/{l+=$NF}
        /vllm:request_success_total.*finished_reason="error"/{e+=$NF}
        END{printf "%.0f %.0f %.0f %.0f %.0f",a+0,w+0,s+0,l+0,e+0}' <<<"$metrics")"
      printf '  vLLM :%s | %s active / %s waiting | %s completed | %s length | %s errors\n' \
        "$subagent_port" "$active" "$waiting" "$completed" "$length" "$errors"
    else
      printf '  vLLM :%s unavailable\n' "$subagent_port"
    fi
  done

  if [[ -z "$FIXED_RUN_ID" ]]; then
    printf '\nFollowing newest run automatically; pass a run ID to pin one.\n'
  fi
  printf 'Refresh every %ss — Ctrl-C to exit\n' "$INTERVAL"
  [[ "$ONCE" == 1 ]] && exit 0
  sleep "$INTERVAL"
done
