"""Shared helpers for agent execution loops."""

from __future__ import annotations

import json
from pathlib import Path

from ..config import MediaConfig, ModelConfig
from ..models.content import ContentBlock, TextBlock, ToolResultBlock
from ..models.task import TaskDefinition
from ..models.trace import MediaLoad
from ..trace.writer import TraceWriter
from .media_loader import collect_media_references, load_media_from_ref, model_supports_modality, to_content_block


def log(msg: str) -> None:
    """Print a log line and flush immediately."""
    print(msg, flush=True)


def brief(d: dict, max_len: int = 80) -> str:
    """Compact one-line summary of a dict for logging."""
    s = json.dumps(d, ensure_ascii=False)
    return s if len(s) <= max_len else s[:max_len] + "..."


def make_local_tool_result(tool_use, text: str, is_error: bool = False) -> ToolResultBlock:
    """Create a ToolResultBlock for a locally dispatched agent tool."""
    return ToolResultBlock(
        tool_use_id=tool_use.id,
        content=[TextBlock(text=text)],
        is_error=is_error,
    )


def cap_conversation_images(messages: list, max_images: int) -> int:
    """Drop earliest image blocks when total exceeds max_images."""
    if max_images <= 0:
        return 0

    total_images = sum(
        1 for msg in messages for b in msg.content if b.type == "image"
    )
    if total_images <= max_images:
        return 0

    protected = sum(
        1 for msg in messages[:2] for b in msg.content if b.type == "image"
    )
    allowed = max(0, max_images - protected)

    positions: list[tuple[int, int]] = []
    for mi in range(2, len(messages)):
        for bi, block in enumerate(messages[mi].content):
            if block.type == "image":
                positions.append((mi, bi))

    if len(positions) <= allowed:
        return 0

    n_drop = len(positions) - allowed
    for mi, bi in positions[:n_drop]:
        messages[mi].content[bi] = TextBlock(
            text="[Image dropped: conversation image limit reached]"
        )
    return n_drop


def strip_old_turn_images(messages: list, keep_recent_turns: int = 3) -> int:
    """Strip ImageBlocks from messages older than keep_recent_turns assistant turns."""
    if keep_recent_turns <= 0:
        return 0

    assistant_indices = [
        i for i, msg in enumerate(messages)
        if msg.role == "assistant"
    ]

    if len(assistant_indices) <= keep_recent_turns:
        return 0

    cutoff_idx = assistant_indices[-keep_recent_turns]

    n_stripped = 0
    for i in range(cutoff_idx):
        msg = messages[i]
        new_content = [b for b in msg.content if b.type != "image"]
        removed = len(msg.content) - len(new_content)
        if removed:
            msg.content = new_content
            n_stripped += removed

    return n_stripped


def build_initial_user_content(
    task: TaskDefinition,
    *,
    trace_id: str,
    writer: TraceWriter,
    model_cfg: ModelConfig | None,
    media_cfg: MediaConfig | None,
) -> list[ContentBlock]:
    content: list[ContentBlock] = [TextBlock(text=task.prompt.text)]
    if media_cfg is not None and not media_cfg.enabled:
        return content

    cfg = media_cfg or MediaConfig()
    model = model_cfg or ModelConfig()
    refs = collect_media_references(task.prompt.text, task.prompt.attachments)
    if not refs:
        return content

    workspace_root = Path.cwd()
    task_dir = Path(task.task_file).parent if task.task_file else None
    for idx, ref in enumerate(refs):
        ref_modality = "image"
        if ref.mime_type:
            if ref.mime_type.startswith("audio/"):
                ref_modality = "audio"
            elif ref.mime_type.startswith("video/"):
                ref_modality = "video"
            elif ref.mime_type.startswith("text/") or ref.mime_type in {"application/json", "application/xml"}:
                ref_modality = "document"
        if idx >= cfg.max_files:
            writer.write_event(MediaLoad(
                trace_id=trace_id,
                modality=ref_modality,  # type: ignore[arg-type]
                source_path=ref.raw_path,
                mime_type=ref.mime_type or "",
                size_bytes=0,
                sha256="",
                status="skipped",
                note=f"exceeds max_files={cfg.max_files}",
            ))
            continue
        try:
            loaded = load_media_from_ref(
                ref,
                workspace_root=workspace_root,
                task_dir=task_dir,
                max_bytes=cfg.max_bytes_per_file,
                image_max_dimension=cfg.image_max_dimension,
            )
            if not model_supports_modality(model.input_modalities, loaded.modality):
                writer.write_event(MediaLoad(
                    trace_id=trace_id,
                    modality=loaded.modality,  # type: ignore[arg-type]
                    source_path=loaded.source_path,
                    mime_type=loaded.mime_type,
                    size_bytes=loaded.size_bytes,
                    sha256=loaded.sha256,
                    status="skipped",
                    note=f"model does not support modality: {loaded.modality}",
                ))
                if cfg.strict_mode:
                    raise ValueError(f"Model {model.model_id} does not support {loaded.modality} input")
                continue
            content.append(to_content_block(loaded))
            writer.write_event(MediaLoad(
                trace_id=trace_id,
                modality=loaded.modality,  # type: ignore[arg-type]
                source_path=loaded.source_path,
                mime_type=loaded.mime_type,
                size_bytes=loaded.size_bytes,
                sha256=loaded.sha256,
                status="loaded",
                note=ref.source,
            ))
        except Exception as exc:
            writer.write_event(MediaLoad(
                trace_id=trace_id,
                modality=ref_modality,  # type: ignore[arg-type]
                source_path=ref.raw_path,
                mime_type=ref.mime_type or "",
                size_bytes=0,
                sha256="",
                status="error",
                note=str(exc),
            ))
            if cfg.strict_mode:
                raise
    return content
