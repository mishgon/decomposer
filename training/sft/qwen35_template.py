from __future__ import annotations


class UnsupportedQwen35TemplateError(ValueError):
    """Raised when the installed Qwen3.5 template cannot be patched safely."""


def _replace_once(template: str, old: str, new: str, description: str) -> str:
    count = template.count(old)
    if count != 1:
        raise UnsupportedQwen35TemplateError(
            f"Expected exactly one {description} in the Qwen3.5 chat template; "
            f"found {count}. The upstream template changed and the assistant-mask "
            "patch must be reviewed."
        )
    return template.replace(old, new, 1)


def build_qwen35_training_template(canonical_template: str) -> str:
    """Mark complete Qwen3.5 assistant turns without changing rendered text.

    Qwen renders environment-owned tool responses in separate ``tool`` branches,
    so one guarded generation region can cover each complete assistant branch.
    """
    if not isinstance(canonical_template, str) or not canonical_template.strip():
        raise UnsupportedQwen35TemplateError("Qwen3.5 has no canonical chat template.")
    if (
        "{% generation" in canonical_template
        or "{% endgeneration" in canonical_template
    ):
        raise UnsupportedQwen35TemplateError(
            "The canonical Qwen3.5 template already contains generation regions; "
            "review whether a custom training template is still necessary."
        )

    assistant_start = '{%- elif message.role == "assistant" %}'
    template = _replace_once(
        canonical_template,
        assistant_start,
        assistant_start + "\n        {%- generation -%}",
        "assistant branch",
    )
    assistant_end = (
        "        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}"
    )
    template = _replace_once(
        template,
        assistant_end,
        "        {{- '<|im_end|>\\n' }}\n"
        "        {%- endgeneration -%}\n"
        '    {%- elif message.role == "tool" %}',
        "assistant branch terminator",
    )
    if template == canonical_template:
        raise AssertionError("Qwen3.5 training-template patch made no changes.")
    return template
