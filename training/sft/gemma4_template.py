from __future__ import annotations


class UnsupportedGemma4TemplateError(ValueError):
    """Raised when the installed Gemma-4 template cannot be patched safely."""


def _replace_once(template: str, old: str, new: str, description: str) -> str:
    count = template.count(old)
    if count != 1:
        raise UnsupportedGemma4TemplateError(
            f"Expected exactly one {description} block in the Gemma-4 chat template; "
            f"found {count}. The upstream template changed and the assistant-mask "
            "patch must be reviewed."
        )
    return template.replace(old, new, 1)


def build_gemma4_training_template(canonical_template: str) -> str:
    """Add assistant generation regions without changing rendered Gemma-4 text.

    The canonical Gemma-4 template renders tool responses inside the surrounding
    model turn. A single generation region around the complete assistant turn
    would therefore train on environment-owned tool output. This patch marks the
    disjoint model-owned regions instead: reasoning, function calls, visible
    assistant content, and the model's end-of-turn token.

    Every replacement is deliberately guarded. If Google changes the canonical
    template, preprocessing fails instead of silently producing incorrect labels.
    """
    if not isinstance(canonical_template, str) or not canonical_template.strip():
        raise UnsupportedGemma4TemplateError("Gemma-4 has no canonical chat template.")
    if "{% generation" in canonical_template or "{% endgeneration" in canonical_template:
        raise UnsupportedGemma4TemplateError(
            "The canonical Gemma-4 template already contains generation regions; "
            "review whether a custom training template is still necessary."
        )

    template = canonical_template

    reasoning_output = (
        "        {{- '<|channel>thought\\n' + thinking_text + '\\n<channel|>' -}}"
    )
    template = _replace_once(
        template,
        reasoning_output,
        "        {%- generation -%}"
        + reasoning_output.strip()
        + "{%- endgeneration -%}",
        "reasoning output",
    )

    tool_call_start = (
        "                    {{- '<|tool_call>call:' + function['name'] + '{' -}}"
    )
    template = _replace_once(
        template,
        tool_call_start,
        "                    {%- generation -%}"
        + tool_call_start.strip(),
        "tool-call start",
    )
    tool_call_end = "                    {{- '}<tool_call|>' -}}"
    template = _replace_once(
        template,
        tool_call_end,
        tool_call_end + "{%- endgeneration -%}",
        "tool-call end",
    )

    captured_content = "            {{- captured_content -}}"
    generated_captured_content = """            {%- if role == 'model' -%}
                {%- generation -%}{{- captured_content -}}{%- endgeneration -%}
            {%- else -%}
                {{- captured_content -}}
            {%- endif -%}"""
    template = _replace_once(
        template,
        captured_content,
        generated_captured_content,
        "captured message content",
    )

    turn_close = """        {%- elif not (ns_tr_out.flag and not has_content and not next_nt.found) -%}
            {{- '<turn|>\\n' -}}
        {%- endif -%}"""
    generated_turn_close = """        {%- elif not (ns_tr_out.flag and not has_content and not next_nt.found) -%}
            {%- if role == 'model' -%}
                {%- generation -%}{{- '<turn|>\\n' -}}{%- endgeneration -%}
            {%- else -%}
                {{- '<turn|>\\n' -}}
            {%- endif -%}
        {%- endif -%}"""
    template = _replace_once(
        template,
        turn_close,
        generated_turn_close,
        "message turn-close",
    )

    if template == canonical_template:
        raise AssertionError("Gemma-4 training-template patch made no changes.")
    return template
