#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib==3.11.2", "numpy==2.5.4", "reportlab==5.1.0",
#   "tensorboard==2.20.0", "PyYAML==6.0.3",
# ]
# ///
"""Render the standard three-page OPD report from a local run directory.

Usage: ./scripts/pdf-report.py /path/to/run [--output /path/to/report.pdf]
Defaults to RUN/opd-report.pdf. Run on the experiment host or on a copied run;
no ClearML credentials, GPU, repository imports or network API calls are needed.
Keep the bundled scripts/report-assets directory beside this script.
The first uv invocation installs the pinned plotting dependencies.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import yaml


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def version(weights):
    value = weights.get("global_steps")
    if value is None or weights.get("min_global_steps", value) != weights.get("max_global_steps", value):
        return None
    return int(value)


def load_run(run):
    train = run / "train" if (run / "train").is_dir() else run
    series = defaultdict(list)
    for directory in sorted({p.parent for p in (train / "tensorboard").rglob("events.out.tfevents.*")}):
        events = EventAccumulator(str(directory), size_guidance={"scalars": 0}).Reload()
        for tag in events.Tags()["scalars"]:
            series[tag].extend(
                {"step": p.step, "value": p.value, "time": p.wall_time} for p in events.Scalars(tag)
            )
    # Resume files can contain the same step. Keep the most recent wall time.
    for tag, points in series.items():
        latest = {p["step"]: p for p in sorted(points, key=lambda p: p["time"])}
        series[tag] = [latest[step] for step in sorted(latest)]
    if not series.get("training/global_step"):
        raise ValueError(f"No training scalars found in {train / 'tensorboard'}")
    config_path = train / "hydra/.hydra/config.yaml"
    config = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
    local = read_json(run / "local-run.json")
    metadata = local | read_json(train / "run.json")
    if local.get("started_at") is not None:
        metadata["started_at"] = local["started_at"]
    manifest = read_json(train / "data/manifest.json")
    rows = defaultdict(lambda: {"episodes": 0, "tokens": 0, "nll_sum": 0.0})
    panels = defaultdict(dict)
    panel_times = {}
    sources = set()

    def record_result(source, step, task_id, result, path):
        # A retry can leave multiple results for the same task and checkpoint.
        key = (source, step, str(task_id))
        finished = result.get("finished_at") or path.stat().st_mtime
        if finished >= panel_times.get(key, 0):
            panels[(source, step)][str(task_id)] = result
            panel_times[key] = finished

    task_ids = set()
    mixed = 0
    for path in sorted((train / "episodes").glob("*/trajectory.json")):
        d = read_json(path)
        source = d.get("data_source", "")
        sources.add(source)
        task_ids.add(str(d.get("task_id", "")))
        v = version(d.get("weight_versions", {}))
        result = d.get("result")
        if source.endswith("/train"):
            if v is None:
                mixed += 1
                continue
            logprobs, mask = d.get("response_logprobs", []), d.get("response_mask", [])
            if len(logprobs) != len(mask):
                raise ValueError(f"Token/logprob alignment mismatch: {path}")
            tokens = [p for p, m in zip(logprobs, mask) if m == 1]
            if not all(math.isfinite(p) for p in tokens):
                raise ValueError(f"Non-finite student logprobs: {path}")
            if tokens:
                rows[v]["episodes"] += 1
                rows[v]["tokens"] += len(tokens)
                rows[v]["nll_sum"] -= sum(tokens)
        elif v is not None and result:
            record_result(source, v, d["task_id"], result, path)
    for path in sorted((train / "workplace-episodes").glob("*/result.json")):
        calls = path.with_name("policy_calls.jsonl")
        if not calls.exists():
            continue
        versions = {
            version(json.loads(line).get("extra_fields") or {})
            for line in calls.read_text().splitlines()
            if line.strip()
        }
        if len(versions) == 1 and None not in versions:
            result = read_json(path)
            record_result("workplace-assistant/validation20", versions.pop(), result["task_id"], result, path)
    gym = (
        "wideseek"
        if (
            any(t.startswith("val-core/wideseek/") for t in series)
            or any(source.startswith("wideseek/") for source in sources)
        )
        else "synth"
    )
    depth = gym == "wideseek" and (
        manifest.get("source") == "depth" or bool(task_ids) and all(t.startswith("depth-") for t in task_ids)
    )
    checkpoint = train / "checkpoints/latest_checkpointed_iteration.txt"
    return {
        "run": run.name,
        "series": dict(series),
        "config": config,
        "run_metadata": metadata,
        "manifest": manifest,
        "gym": gym,
        "depth": depth,
        "panels": panels,
        "checkpoint": int(checkpoint.read_text().strip()) if checkpoint.exists() else None,
        "captured_at": max(p["time"] for points in series.values() for p in points),
        "proxy": {
            "rows": [
                dict(version=v, **r, mean_nll=r["nll_sum"] / r["tokens"]) for v, r in sorted(rows.items())
            ],
            "excluded_mixed_version_episodes": mixed,
        },
    }


def timeout(result):
    return result.get("status") == "timeout" or "TimeoutError" in str(result.get("error", ""))


def validation_tags(series, gym):
    tags = []
    for source in (f"{gym}/train_probe", f"{gym}/heldout", "workplace-assistant/validation20"):
        prefix = f"val-core/{source}/reward/mean@"
        candidates = [tag for tag in series if tag.startswith(prefix)]
        tags.append(max(candidates, key=lambda tag: int(tag.split("@")[-1])) if candidates else prefix + "1")
    return tags


def planned_updates(d):
    metadata, config, manifest = d["run_metadata"], d["config"], d["manifest"]
    if metadata.get("optimizer_updates") is not None:
        return metadata["optimizer_updates"]
    groups = config.get("rollout", {}).get("total_rollout_steps")
    if manifest and config:
        available = (
            manifest["train_tasks"]
            * manifest["attempts_per_task_per_epoch"]
            * config["trainer"]["total_epochs"]
        )
        groups = min(groups, available) if groups is not None else available
    if groups is None:
        return None
    asynchronous = config["async_training"]
    batch = (
        config["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"]
        * asynchronous["require_batches"]
        * asynchronous["trigger_parameter_sync_step"]
    )
    return groups // batch


def render_report(d, output):
    s, proxy, metadata = d["series"], d["proxy"], d["run_metadata"]
    gym = d["gym"]
    name = "WideSeek depth" if d["depth"] else "WideSeek" if gym == "wideseek" else "Synth"
    model = Path(metadata.get("model", "Starting model not recorded")).name
    data_config = d["config"].get("data", {})
    response_budget = data_config.get("max_response_length")
    prompt_budget = data_config.get("max_prompt_length")
    total_budget = (
        response_budget + prompt_budget if response_budget is not None and prompt_budget is not None else "?"
    )

    def fit_text(text, font, size, x, y, width=None):
        width = width or W - 60
        measured = pdf.stringWidth(text, font, size)
        pdf.setFont(font, min(size, size * width / max(measured, 1)))
        pdf.drawString(x, y, text)

    # AIJ 2026: Inter typography and the AIRI palette supplied with the demo guide.
    assets = Path(__file__).resolve().parent / "report-assets"
    for weight in ("Regular", "Medium", "SemiBold"):
        font = assets / f"Inter-{weight}.ttf"
        font_manager.fontManager.addfont(str(font))
        pdfmetrics.registerFont(TTFont(f"Inter-{weight}", str(font)))
    C = {
        "blue": "#1394C1",
        "green": "#005A60",
        "brown": "#8E5F44",
        "gray": "#909393",
        "sand": "#D1A684",
    }
    plt.rcParams.update(
        {
            "font.family": "Inter",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.edgecolor": "#909393",
            "grid.color": "#EFF0EF",
            "axes.titleweight": "medium",
            "font.weight": "normal",
            "savefig.facecolor": "white",
        }
    )

    def xy(tag):
        p = s.get(tag, [])
        return np.array([r["step"] for r in p]), np.array([r["value"] for r in p])

    def base(title, ylabel, xlabel="Logged step"):
        fig, ax = plt.subplots(figsize=(5.6, 2.85))
        ax.set(title=title, ylabel=ylabel, xlabel=xlabel)
        ax.grid(axis="y", alpha=0.8)
        ax.set_axisbelow(True)
        return fig, ax

    def line(ax, tag, label, color, **kw):
        x, y = xy(tag)
        ax.plot(x, y, label=label if len(y) else f"{label} (not recorded)", color=color, **kw)
        return x, y

    def img(fig):
        if not any(ax.has_data() for ax in fig.axes):
            fig.axes[0].text(
                0.5,
                0.5,
                "Not recorded",
                ha="center",
                va="center",
                transform=fig.axes[0].transAxes,
                color=C["gray"],
            )
        fig.tight_layout(pad=1.0)
        b = io.BytesIO()
        fig.savefig(b, format="png", dpi=210)
        plt.close(fig)
        b.seek(0)
        return ImageReader(b)

    W, H = landscape(A4)
    pdf = canvas.Canvas(str(output), pagesize=(W, H), invariant=1)
    pdf.setTitle(f"{name} OPD report: {d['run']}")
    pdf.setAuthor("Decomposer")
    when = datetime.fromtimestamp(d["captured_at"], timezone.utc).strftime("%d %b %Y, %H:%M UTC")

    def page(n, title, subtitle, notes):
        pdf.setFillColor(HexColor("#2FBEAD"))
        pdf.rect(30, H - 19, 32, 3, fill=1, stroke=0)
        pdf.setFillColor(HexColor("#000000"))
        fit_text(title, "Inter-Medium", 22, 30, H - 42, W - 178)
        fit_text(subtitle, "Inter-Regular", 10, 30, H - 61, W - 178)
        pdf.drawImage(str(assets / "airi-logo.png"), W - 125, H - 65,
                      width=95, height=48, preserveAspectRatio=True, mask="auto")
        pdf.setStrokeColor(HexColor("#84E0D5"))
        pdf.line(30, 66, W - 30, 66)
        pdf.setFont("Inter-Regular", 8)
        for i, t in enumerate(notes):
            fit_text(t, "Inter-Regular", 8, 30, 53 - i * 11)
        pdf.setFillColor(HexColor("#909393"))
        fit_text(f"{d['run']} | latest event {when}", "Inter-Regular", 7, 30, 15, W - 90)
        pdf.drawRightString(W - 30, 15, f"{n} / 3")

    def place(fig, col, row):
        width = (W - 70) / 2
        height = 211
        pdf.drawImage(
            img(fig),
            30 + col * (width + 10),
            H - 78 - (row + 1) * height - row * 8,
            width=width,
            height=height,
            mask="auto",
        )

    tags = validation_tags(s, gym)
    latest_val = max((int(xy(t)[0][-1]) for t in tags if len(xy(t)[0])), default=0)
    held_x, held_y = xy(tags[1])
    held_note = (
        f"Held-out: {held_y[0]:.1%} initial, {max(held_y):.1%} peak (step {held_x[np.argmax(held_y)]:.0f}), {held_y[-1]:.1%} latest."
        if len(held_y)
        else "Held-out validation not recorded."
    )
    validation_notes = []
    for tag, label in zip(tags, ["train probes", "held-out", "Workplace"]):
        x, _ = xy(tag)
        results = (
            d["panels"].get(
                (
                    tag.removeprefix("val-core/").split("/reward/")[0],
                    int(x[-1]),
                ),
                {},
            )
            if len(x)
            else {}
        )
        validation_notes.append(
            f"{label}: {sum(timeout(r) for r in results.values())}/{len(results)} timeouts"
            if results
            else f"{label}: traces unavailable"
        )
    reward_kind = (
        "binary QA accuracy"
        if d["depth"]
        else "native item-F1"
        if gym == "wideseek"
        else "native synthetic reward"
    )
    status = metadata.get("status", "unknown")
    subtitle = f"{model} | {d['manifest'].get('train_tasks', '?')} training tasks | {total_budget} total tokens | {status}"
    page(
        1,
        f"{name} OPD: validation through step {latest_val}",
        subtitle,
        [
            f"{name} uses {reward_kind}; Workplace uses binary task success. Curves use the logged native validation means.",
            held_note + " Training rollouts use changing tasks; fixed-panel scores are reported separately.",
            "Latest validation traces (last attempt/task): "
            + "; ".join(validation_notes)
            + ". Timeouts are included in logged rewards.",
        ],
    )
    f, a = base("Fixed-panel validation", "Mean reward / success rate", "Validation step")
    for tag, label, col in zip(
        tags,
        ["Train probes", "Held-out", "Workplace OOD"],
        [C["blue"], C["green"], C["brown"]],
    ):
        x, y = line(a, tag, label, col, marker="o", lw=2)
        if len(y):
            a.annotate(
                f"{y[-1]:.3f}",
                (x[-1], y[-1]),
                xytext=(5, -10 if "train_probe" in tag else 8),
                textcoords="offset points",
                color="black",
                fontsize=9,
            )
    a.set_ylim(-0.04, 1.04)
    a.set_xlim(-1, latest_val + 8)
    a.set_xticks(range(0, latest_val + 1, 8))
    a.legend(loc="lower left")
    place(f, 0, 0)
    common = set.intersection(*(set(xy(t)[0]) for t in tags))
    first, last = (min(common), max(common)) if common else (None, None)
    f, a = base(
        f"Change from step {first:.0f} to step {last:.0f}" if common else "Change between validation panels",
        "Change in mean score",
        xlabel="",
    )
    if common:
        delta = [dict(zip(*xy(t)))[last] - dict(zip(*xy(t)))[first] for t in tags]
        a.bar(
            ["Train probes", "Held-out", "Workplace OOD"],
            delta,
            color=[C["blue"], C["green"], C["brown"]],
            width=0.55,
        )
        a.axhline(0, color=C["gray"], lw=0.8)
        a.set_ylim(min(-0.15, min(delta) - 0.15), max(0.15, max(delta) + 0.15))
        for i, v in enumerate(delta):
            a.text(
                i,
                v + (0.035 if v >= 0 else -0.07),
                f"{v:+.3f}",
                ha="center",
                weight="semibold",
            )
    place(f, 1, 0)
    f, a = base("Training rollout reward", "Native batch mean reward")
    x, y = line(
        a,
        "critic/rewards/mean",
        "Raw batch (changing tasks)",
        C["blue"],
        alpha=0.32,
        lw=1,
    )
    if len(y) >= 5:
        smooth = np.convolve(y, np.ones(5) / 5, mode="valid")
        a.plot(x[4:], smooth, color=C["blue"], lw=2, label="Trailing 5 batches")
    a.set_ylim(0, 1)
    a.legend()
    place(f, 0, 1)
    f, a = base("Rollout length and token pressure", "Response tokens")
    x, y = line(a, "response_length/mean", "Batch mean", C["green"])
    lo = dict(zip(*xy("response_length/min")))
    hi = dict(zip(*xy("response_length/max")))
    shared = [v for v in x if v in lo and v in hi]
    if shared:
        a.fill_between(
            shared,
            [lo[v] for v in shared],
            [hi[v] for v in shared],
            color=C["green"],
            alpha=0.13,
            label="Batch min-max",
        )
    if response_budget is not None and prompt_budget is not None:
        a.axhline(
            response_budget,
            color=C["brown"],
            ls="--",
            lw=1,
            label=f"{response_budget:,} response budget ({total_budget:,} total)",
        )
    a.legend(loc="upper left")
    place(f, 1, 1)
    pdf.showPage()

    page(
        2,
        "How the policy is changing",
        f"{model}; loss and surprisal describe updates, not task-solving quality.",
        [
            "Signed and absolute distillation metrics are shown below. Task rewards are reported separately on page 1.",
            f"Proxy uses {sum(r['episodes'] for r in proxy['rows'])} train rollouts; mixed-version episodes excluded: {proxy['excluded_mixed_version_episodes']}. KL compares rollout/update policies, not teacher quality.",
            "Entropy proxy: token-weighted -log p of sampled student tokens; excludes observations. Sampling filters/temperature prevent treating it as full entropy.",
        ],
    )
    f, a = base("Distillation objective", "Reported loss")
    line(a, "actor/distillation/loss", "Signed distillation loss", C["blue"], lw=1.6)
    line(a, "actor/distillation/abs_loss", "Absolute-loss metric", C["sand"], lw=1.4)
    a.axhline(0, color=C["gray"], lw=0.7)
    a.legend()
    place(f, 0, 0)
    f, a = base("Gradient magnitude", "Reported gradient norm")
    line(a, "actor/grad_norm", "Gradient norm", C["green"], lw=1.6)
    a.set_ylim(bottom=0)
    place(f, 1, 0)
    f, a = base("Update size and clipping", "KL estimate / clipped fraction")
    line(a, "actor/distillation/ppo_kl", "Sampled PPO KL", C["blue"], lw=1.4)
    line(
        a,
        "actor/distillation/pg_clipfrac",
        "Distillation clipped fraction",
        C["brown"],
        lw=1.4,
    )
    a.axhline(0, color=C["gray"], lw=0.7)
    a.legend()
    place(f, 0, 1)
    f, a = base(
        "Entropy proxy: sampled-token surprisal",
        "Mean -log p(token), nats",
        "Rollout policy version",
    )
    rows = proxy["rows"]
    a.plot(
        [r["version"] for r in rows],
        [r["mean_nll"] for r in rows],
        color=C["sand"],
        lw=1.7,
        marker=".",
        ms=4,
    )
    a.set_ylim(bottom=0)
    a.text(
        0.03,
        0.92,
        "Full-distribution entropy is not logged."
        if rows
        else "Student token logprobs / versions not recorded.",
        transform=a.transAxes,
        fontsize=8,
        color=C["gray"],
    )
    place(f, 1, 1)
    pdf.showPage()

    points = s["training/global_step"]
    start = metadata.get("started_at", points[0]["time"])
    end = metadata.get("finished_at") or d["captured_at"]
    minutes = max(0, round((end - start) / 60))
    checkpoint = d["checkpoint"]
    page(
        3,
        "Where wall-clock time goes",
        f"{status.capitalize()} | {minutes // 60}h {minutes % 60:02d}m since recorded launch, including initialization and validation.",
        [
            "Observed cadence uses consecutive log wall times; asynchronous components can overlap. X axes use event steps unless specified.",
            f"{len(points)} optimizer counters logged: {points[0]['value']:g}-{points[-1]['value']:g}; event steps {points[0]['step']}-{points[-1]['step']}. Latest saved checkpoint: {checkpoint if checkpoint is not None else 'unknown'}.",
            "Observed maxima: "
            + "; ".join(
                f"{label} {max(xy(tag)[1]):.1%}" if len(xy(tag)[1]) else f"{label} unavailable"
                for tag, label in [
                    ("response_length/clip_ratio", "response clipping"),
                    ("response/aborted_ratio", "aborted responses"),
                ]
            )
            + ".",
        ],
    )
    f, a = base("Per-step timing", "Seconds")
    p = points
    x = np.array([r["step"] for r in p])
    t = np.array([r["time"] for r in p])
    a.plot(
        x[1:],
        np.diff(t),
        color=C["gray"],
        alpha=0.65,
        label="Observed log-to-log interval",
    )
    line(a, "timing_s/update_actor", "Actor update", C["blue"], lw=1.3)
    line(a, "timing_s/save_checkpoint", "Checkpoint save", C["sand"], lw=1.3)
    a.legend(fontsize=7)
    a.set_ylim(bottom=0)
    place(f, 0, 0)
    f, a = base(
        "Time spent idle",
        "Fraction of interval",
        xlabel="Logged step / rollout version",
    )
    line(a, "fully_async/rollouter/idle_ratio", "Rollouter idle", C["brown"], lw=1.4)
    line(a, "fully_async/trainer/idle_ratio", "Trainer idle", C["blue"], lw=1.4)
    a.set_ylim(-0.02, 1.02)
    a.legend()
    place(f, 1, 0)
    f, a = base("Recorded training progress", "Optimizer counter", "Hours since recorded launch")
    a.plot((t - start) / 3600, [r["value"] for r in p], color=C["blue"], lw=2)
    if status == "completed" and checkpoint is not None:
        a.scatter(
            [(end - start) / 3600],
            [checkpoint],
            color=C["green"],
            s=25,
            label=f"Checkpoint {checkpoint} / completed",
        )
    planned = planned_updates(d)
    if planned is not None:
        a.axhline(planned, color=C["gray"], ls="--", label=f"Planned {planned} updates")
    a.set_ylim(bottom=0)
    a.legend()
    place(f, 0, 1)
    f, a = base("Typical time components", "Median seconds per logged step", xlabel="")
    keys = [
        "timing_s/update_actor",
        "timing_s/timing_s/param_sync",
        "timing_s/save_checkpoint",
    ]
    v = [float(np.median(xy(k)[1])) if len(xy(k)[1]) else float("nan") for k in keys]
    a.bar(
        ["Actor update", "Parameter sync", "Checkpoint"],
        v,
        color=[C["blue"], C["green"], C["sand"]],
        width=0.55,
    )
    for i, z in enumerate(v):
        a.text(
            i,
            z + 3 if math.isfinite(z) else 3,
            f"{z:.0f}s" if math.isfinite(z) else "N/A",
            ha="center",
            weight="semibold",
        )
    a.set_ylim(0, max([z for z in v if math.isfinite(z)] + [10]) * 1.25)
    place(f, 1, 1)
    pdf.save()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "run",
        type=Path,
        help="Run directory containing train/tensorboard and saved episodes",
    )
    parser.add_argument("--output", type=Path, help="PDF destination (default: RUN/opd-report.pdf)")
    args = parser.parse_args()
    output = args.output or args.run / "opd-report.pdf"
    try:
        data = load_run(args.run)
        output.parent.mkdir(parents=True, exist_ok=True)
        render_report(data, output)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Cannot build OPD report: {error}\n")
    print(output.resolve())


if __name__ == "__main__":
    main()
