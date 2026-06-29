"""Load config.yaml with env-var expansion and provider auto-selection."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


_ENV_RE = re.compile(r"\$\{(\w+)\}")

# Search order: CWD -> project root (where pyproject.toml lives)
_SEARCH_PATHS = [
    Path.cwd() / "config.yaml",
    Path(__file__).resolve().parent.parent.parent / "config.yaml",
]
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DOTENV_PATHS = [
    Path.cwd() / ".env",
    _PROJECT_ROOT / ".env",
]


def _expand_env(value: str) -> str | None:
    """Replace ${VAR} with os.environ[VAR]. Returns None if var is unset."""
    m = _ENV_RE.fullmatch(value.strip())
    if m:
        return os.environ.get(m.group(1))
    return value


def _walk_expand(obj):
    """Recursively expand ${ENV} references in string values."""
    if isinstance(obj, str):
        return _expand_env(obj)
    if isinstance(obj, dict):
        return {k: _walk_expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_expand(v) for v in obj]
    return obj


def _load_dotenv() -> None:
    """Load .env files without overriding already-exported environment values."""
    for dotenv_path in _DOTENV_PATHS:
        if dotenv_path.exists():
            load_dotenv(dotenv_path, override=False)


def _env_first(*names: str) -> str | None:
    """Return the first non-empty environment variable value."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _env_flag(name: str) -> bool:
    """Return True for common truthy environment flag values."""
    value = os.environ.get(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


class ModelConfig(BaseModel):
    api_key: str | None = None
    base_url: str | None = None
    model_id: str = "anthropic/claude-opus-4-6"
    input_modalities: list[str] = Field(default_factory=lambda: ["text"])
    system_prompt_prefix: str | None = None
    extra_body: dict | None = None
    reasoning_effort: str | None = None
    context_window: int = 262144
    temperature: float | None = 0.0  # None = don't send temperature param


class JudgeConfig(BaseModel):
    api_key: str | None = None
    base_url: str = "https://openrouter.ai/api/v1"
    model_id: str = "google/gemini-3-flash-preview"
    extra_body: dict | None = None
    enabled: bool = True


class DefaultsConfig(BaseModel):
    trace_dir: str = "traces"
    tasks_dir: str = "tasks"


class SandboxConfig(BaseModel):
    """Configuration for Docker sandbox execution."""

    enabled: bool = False
    image: str = "claw-eval-agent:latest"
    docker_host: str | None = None
    memory_limit: str = "4g"
    cpu_limit: float = 2.0
    sandbox_port: int = 8080
    container_timeout: int = 900
    max_concurrent: int = 10
    enable_browser: bool = True
    enable_shell: bool = True
    enable_file: bool = True


class PromptFilesConfig(BaseModel):
    """Workspace markdown files to inject into system prompt."""

    agents_md: str | None = None
    soul_md: str | None = None
    user_md: str | None = None
    tools_md: str | None = None


class SkillEntry(BaseModel):
    """A skill descriptor shown in the default skills list."""

    name: str
    description: str
    path: str


class SkillsConfig(BaseModel):
    """Skills configuration for prompt composition."""

    default: list[SkillEntry] = Field(default_factory=list)
    load_via_tool_call: bool = True
    read_tool_name: str = "read"


class BehaviorRulesConfig(BaseModel):
    """Behavior-policy text included in system prompt."""

    safety: str = "No independent objective; do not pursue self-preservation, replication, or resource acquisition."
    tool_call_style: str = "For low-risk actions, call tools directly without narration; narrate only for complex tasks."
    reply_tags: str = "Use [[reply_to_current]] to control reply relationship when needed."
    silent_reply: str = "If no reply is needed, output NO_REPLY."
    heartbeat: str = "Heartbeat checks should return HEARTBEAT_OK when no action is needed."


class PromptConfig(BaseModel):
    """Configuration for dynamic system prompt construction."""

    enabled: bool = True
    strict_file_check: bool = False
    include_tool_schema: bool = True
    files: PromptFilesConfig = PromptFilesConfig()
    behavior_rules: BehaviorRulesConfig = BehaviorRulesConfig()
    skills: SkillsConfig = SkillsConfig()


class MediaConfig(BaseModel):
    """Configuration for media detection and loading from prompts."""

    enabled: bool = True
    strict_mode: bool = False
    max_files: int = 6
    max_bytes_per_file: int = 8 * 1024 * 1024
    image_max_dimension: int = 2048
    # Tool-media injection settings (for ReadMedia / Read with image/PDF)
    inject_tool_media: bool = True
    max_images_per_turn: int = 64
    max_tool_images_total: int = 64
    video_frame_budget: int = 8
    tool_image_quality: int = 60
    tool_image_max_dimension: int = 1280
    max_conversation_images: int = 256
    image_keep_recent_turns: int = 3


class UserAgentModelConfig(BaseModel):
    """LLM configuration for simulated user agent."""
    api_key: str | None = None
    base_url: str = "https://openrouter.ai/api/v1"
    model_id: str = "google/gemini-3-flash-preview"


class DecomposerRunConfig(BaseModel):
    """Limits for hierarchical decomposer runs."""

    max_decomposer_turns: int = 15
    max_delegations: int = 8
    min_delegations_before_final: int = 1
    decomposer_max_output_tokens: int | None = 2048
    executor_max_turns: int | None = None
    executor_max_output_tokens: int | None = 1024
    executor_min_tool_calls: int = 1
    executor_max_environment_tool_calls: int | None = 20
    executor_report_max_tokens: int | None = 512
    executor_report_mode: Literal["strict", "repair", "structured", "structured_repair"] = "strict"
    executor_synthetic_failure_report: bool = True
    executor_prompt_mode: Literal["report_wrapper", "flat_subtask"] = "report_wrapper"
    executor_evidence_mode: Literal["none", "tool_summary"] = "none"
    executor_evidence_max_chars: int = 2000
    manager_valid_tool_guidance: bool = False


class ReActRunConfig(BaseModel):
    """Shared ReAct-loop controls for flat and executor runs."""

    max_turns: int | None = None
    max_environment_tool_calls: int | None = None
    retry_empty_model_response: bool = False
    retry_missing_required_tool: bool = False
    retry_transitional_tool_text: bool = False
    transitional_tool_retry_limit: int = 2
    transitional_tool_phrases: list[str] = Field(default_factory=lambda: [
        "let me",
        "i will",
        "i'll",
        "i need to",
        "need to",
        "next i",
        "now i",
    ])


class VllmConfig(BaseModel):
    """vLLM local inference settings (env-driven; YAML optional)."""

    enabled: bool = False
    wait_for_model: bool = True
    ready_timeout_s: int = 600
    ready_poll_s: float = 5.0
    check_health: bool = True


class Config(BaseModel):
    model: ModelConfig = ModelConfig()
    executor_model: ModelConfig | None = None
    judge: JudgeConfig = JudgeConfig()
    defaults: DefaultsConfig = DefaultsConfig()
    sandbox: SandboxConfig = SandboxConfig()
    prompt: PromptConfig = PromptConfig()
    media: MediaConfig = MediaConfig()
    user_agent_model: UserAgentModelConfig = UserAgentModelConfig()
    decomposer: DecomposerRunConfig = DecomposerRunConfig()
    react: ReActRunConfig = ReActRunConfig()
    vllm: VllmConfig = VllmConfig()
    provider_mode: str = "openrouter"  # vllm | local | openrouter


RoleName = str  # model | judge | user_agent | executor | decomposer


def _vllm_enabled() -> bool:
    if _env_flag("VLLM_ENABLED"):
        return True
    return bool(_env_first("VLLM_BASE_URL"))


def _normalize_vllm_base_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url


def _strip_openrouter_extra_body_payload(extra_body: dict | None) -> dict | None:
    """Drop OpenRouter-only request fields while keeping local vLLM fields."""
    if not extra_body:
        return None
    kept = {
        key: value
        for key, value in extra_body.items()
        if key not in {"reasoning", "provider", "transforms"}
    }
    return kept or None


def _strip_openrouter_extra_body(cfg: Config) -> None:
    if not (
        _env_flag("VLLM_KEEP_EXTRA_BODY")
        or _env_flag("LOCALINFERENCE_KEEP_EXTRA_BODY")
        or _env_flag("LOCAL_INFERENCE_KEEP_EXTRA_BODY")
    ):
        cfg.model.extra_body = _strip_openrouter_extra_body_payload(cfg.model.extra_body)
        if cfg.executor_model is not None:
            cfg.executor_model.extra_body = _strip_openrouter_extra_body_payload(cfg.executor_model.extra_body)


def _apply_vllm_role(
    cfg: Config,
    role: RoleName,
    *,
    model_cfg: ModelConfig | JudgeConfig | UserAgentModelConfig,
    yaml_model_id: str,
) -> None:
    """Apply vLLM base_url/api_key/model_id for one role."""
    role_upper = role.upper()
    base_url = _env_first(f"VLLM_{role_upper}_BASE_URL", "VLLM_BASE_URL")
    if not base_url:
        return
    base_url = _normalize_vllm_base_url(base_url)
    api_key = _env_first(f"VLLM_{role_upper}_API_KEY", "VLLM_API_KEY") or "unused"
    model_id = (
        _env_first(f"VLLM_{role_upper}_MODEL_ID", "VLLM_MODEL_ID")
        or yaml_model_id
    )

    if isinstance(model_cfg, ModelConfig):
        model_cfg.api_key = api_key
        model_cfg.base_url = base_url
        model_cfg.model_id = model_id
    elif isinstance(model_cfg, JudgeConfig):
        model_cfg.api_key = api_key
        model_cfg.base_url = base_url
        model_cfg.model_id = model_id
    elif isinstance(model_cfg, UserAgentModelConfig):
        model_cfg.api_key = api_key
        model_cfg.base_url = base_url
        model_cfg.model_id = model_id


def _apply_vllm_selection(cfg: Config) -> Config | None:
    """Apply vLLM env overrides when VLLM_BASE_URL or VLLM_ENABLED is set."""
    if not _vllm_enabled():
        return None

    cfg.provider_mode = "vllm"
    cfg.vllm.enabled = True
    if _env_first("VLLM_WAIT_FOR_MODEL") is not None:
        cfg.vllm.wait_for_model = _env_flag("VLLM_WAIT_FOR_MODEL")
    timeout = _env_first("VLLM_READY_TIMEOUT_S")
    if timeout:
        cfg.vllm.ready_timeout_s = int(timeout)
    poll = _env_first("VLLM_READY_POLL_S")
    if poll:
        cfg.vllm.ready_poll_s = float(poll)
    if _env_first("VLLM_CHECK_HEALTH") is not None:
        cfg.vllm.check_health = _env_flag("VLLM_CHECK_HEALTH")

    _apply_vllm_role(cfg, "model", model_cfg=cfg.model, yaml_model_id=cfg.model.model_id)
    if cfg.executor_model is not None and _env_first("VLLM_DECOMPOSER_BASE_URL", "VLLM_DECOMPOSER_MODEL_ID"):
        _apply_vllm_role(cfg, "decomposer", model_cfg=cfg.model, yaml_model_id=cfg.model.model_id)

    _apply_vllm_role(cfg, "judge", model_cfg=cfg.judge, yaml_model_id=cfg.judge.model_id)

    _apply_vllm_role(
        cfg,
        "user_agent",
        model_cfg=cfg.user_agent_model,
        yaml_model_id=cfg.user_agent_model.model_id,
    )

    if cfg.executor_model is None:
        cfg.executor_model = ModelConfig()
    _apply_vllm_role(
        cfg,
        "executor",
        model_cfg=cfg.executor_model,
        yaml_model_id=cfg.executor_model.model_id,
    )

    _strip_openrouter_extra_body(cfg)
    return cfg


def resolve_role_endpoint(
    cfg: Config,
    role: RoleName,
    *,
    cli_model_id: str | None = None,
) -> tuple[str | None, str | None, str]:
    """Return (base_url, api_key, model_id) for a role after config resolution."""
    if role in ("model", "decomposer"):
        mc = cfg.model
    elif role == "executor":
        mc = cfg.executor_model or cfg.model
    elif role == "judge":
        base = cfg.judge.base_url
        return base, cfg.judge.api_key, cli_model_id or cfg.judge.model_id
    elif role == "user_agent":
        base = cfg.user_agent_model.base_url
        return base, cfg.user_agent_model.api_key, cfg.user_agent_model.model_id
    else:
        mc = cfg.model

    model_id = cli_model_id or mc.model_id
    return mc.base_url, mc.api_key, model_id


def collect_vllm_wait_targets(
    cfg: Config,
    *,
    roles: list[RoleName],
    cli_overrides: dict[RoleName, str | None] | None = None,
    include_judge: bool = True,
) -> list[tuple[str, str, str]]:
    """Collect unique (base_url, api_key, model_id) tuples for readiness polling."""
    if cfg.provider_mode != "vllm" or not cfg.vllm.wait_for_model:
        return []

    overrides = cli_overrides or {}
    seen: set[tuple[str, str]] = set()
    targets: list[tuple[str, str, str]] = []

    effective_roles = list(roles)
    if include_judge and "judge" not in effective_roles and cfg.judge.enabled:
        effective_roles.append("judge")

    for role in effective_roles:
        if role == "judge" and not include_judge:
            continue
        base_url, api_key, model_id = resolve_role_endpoint(
            cfg, role, cli_model_id=overrides.get(role),
        )
        if not base_url or not model_id:
            continue
        key = (base_url, model_id)
        if key in seen:
            continue
        seen.add(key)
        targets.append((base_url, api_key or "unused", model_id))

    return targets


def _apply_provider_auto_selection(cfg: Config) -> Config:
    """Prefer vLLM, then local gateway, then OpenRouter.

    vLLM (priority 1): VLLM_BASE_URL or VLLM_ENABLED=1
    Local gateway (priority 2): LOCALINFERENCE_API_KEY + LOCALINFERENCE_BASE_URL
    OpenRouter (priority 3): OPENROUTER_API_KEY
    """
    vllm_cfg = _apply_vllm_selection(cfg)
    if vllm_cfg is not None:
        return vllm_cfg

    local_api_key = _env_first("LOCALINFERENCE_API_KEY", "LOCAL_INFERENCE_API_KEY")
    local_base_url = _env_first("LOCALINFERENCE_BASE_URL", "LOCAL_INFERENCE_BASE_URL")
    if local_api_key and local_base_url:
        cfg.provider_mode = "local"
        cfg.model.api_key = local_api_key
        cfg.model.base_url = local_base_url
        cfg.model.model_id = _env_first(
            "LOCALINFERENCE_MODEL_ID",
            "LOCAL_INFERENCE_MODEL_ID",
        ) or cfg.model.model_id
        if not (
            _env_flag("LOCALINFERENCE_KEEP_EXTRA_BODY")
            or _env_flag("LOCAL_INFERENCE_KEEP_EXTRA_BODY")
        ):
            cfg.model.extra_body = _strip_openrouter_extra_body_payload(cfg.model.extra_body)

        cfg.judge.api_key = local_api_key
        cfg.judge.base_url = local_base_url
        cfg.judge.model_id = _env_first(
            "LOCALINFERENCE_JUDGE_MODEL_ID",
            "LOCAL_INFERENCE_JUDGE_MODEL_ID",
            "LOCALINFERENCE_MODEL_ID",
            "LOCAL_INFERENCE_MODEL_ID",
        ) or cfg.judge.model_id

        cfg.user_agent_model.api_key = local_api_key
        cfg.user_agent_model.base_url = local_base_url
        cfg.user_agent_model.model_id = _env_first(
            "LOCALINFERENCE_USER_AGENT_MODEL_ID",
            "LOCAL_INFERENCE_USER_AGENT_MODEL_ID",
            "LOCALINFERENCE_MODEL_ID",
            "LOCAL_INFERENCE_MODEL_ID",
        ) or cfg.user_agent_model.model_id

        if cfg.executor_model is None:
            cfg.executor_model = ModelConfig()
        cfg.executor_model.api_key = local_api_key
        cfg.executor_model.base_url = local_base_url
        cfg.executor_model.model_id = _env_first(
            "LOCALINFERENCE_EXECUTOR_MODEL_ID",
            "LOCAL_INFERENCE_EXECUTOR_MODEL_ID",
            "LOCALINFERENCE_MODEL_ID",
            "LOCAL_INFERENCE_MODEL_ID",
        ) or cfg.executor_model.model_id
        if not (
            _env_flag("LOCALINFERENCE_KEEP_EXTRA_BODY")
            or _env_flag("LOCAL_INFERENCE_KEEP_EXTRA_BODY")
        ):
            cfg.executor_model.extra_body = _strip_openrouter_extra_body_payload(cfg.executor_model.extra_body)

        decomposer_model_id = _env_first(
            "LOCALINFERENCE_DECOMPOSER_MODEL_ID",
            "LOCAL_INFERENCE_DECOMPOSER_MODEL_ID",
        )
        if decomposer_model_id:
            cfg.model.model_id = decomposer_model_id

        return cfg

    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_api_key:
        cfg.model.api_key = cfg.model.api_key or openrouter_api_key
        cfg.judge.api_key = cfg.judge.api_key or openrouter_api_key
        cfg.user_agent_model.api_key = cfg.user_agent_model.api_key or openrouter_api_key
    return cfg


def load_config(path: str | Path | None = None) -> Config:
    """Load config from YAML file with ${ENV} expansion.

    Searches config.yaml in CWD then project root if path is not given.
    Returns defaults if no file is found.
    """
    _load_dotenv()

    if path is not None:
        candidates = [Path(path)]
    else:
        candidates = _SEARCH_PATHS

    for p in candidates:
        if p.exists():
            with open(p) as f:
                raw = yaml.safe_load(f) or {}
            expanded = _walk_expand(raw)
            return _apply_provider_auto_selection(Config.model_validate(expanded))

    return _apply_provider_auto_selection(Config())
