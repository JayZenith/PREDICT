"""Render GLYPH's structured agent messages as explicit ChatML."""

from __future__ import annotations

from typing import Any


IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
ARM_A_SYSTEM_PROMPT = (
    "You are a Python coding agent. Use tools when needed. "
    "After FINAL, stop immediately."
)
ARM_B_SYSTEM_PROMPT = (
    "You are a Python coding agent. After apply_patch succeeds, predict before "
    "testing: emit <PREDICTION>OUTCOME</PREDICTION>, then "
    "<DECISION>KEEP</DECISION> to test or <DECISION>REVISE</DECISION> to patch. "
    "OUTCOME is PASS, ASSERTION_FAILURE, RUNTIME_ERROR, SYNTAX_ERROR, TIMEOUT, "
    "or OTHER. After FINAL, stop."
)
DEFAULT_SYSTEM_PROMPT = ARM_A_SYSTEM_PROMPT

GLYPH_CHAT_TEMPLATE = r"""{%- for message in messages %}
{%- set role = message['role'] %}
{%- set content = message['content'] %}
{{- '<|im_start|>' + role + '\n' + content.rstrip() + '\n<|im_end|>\n\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|im_start|>assistant\n' }}
{%- endif %}"""


def _message_value(message: Any, key: str, default: str = "") -> Any:
    if isinstance(message, dict):
        return message.get(key, default)
    value = getattr(message, key, default)
    if value is default and hasattr(message, "model_dump"):
        value = message.model_dump().get(key, default)
    return default if value is None else value


def message_role(message: Any) -> str:
    return str(_message_value(message, "role"))


def message_content(message: Any) -> str:
    return str(_message_value(message, "content"))


def message_tool_call_id(message: Any) -> str:
    return str(_message_value(message, "tool_call_id"))


def render_message(role: str, content: str) -> str:
    return f"{IM_START}{role}\n{content.rstrip()}\n{IM_END}"


def render_messages(messages: list[Any], add_generation_prompt: bool = False) -> str:
    rendered = "".join(
        f"{render_message(message_role(message), message_content(message))}\n\n"
        for message in messages
        if message_role(message)
    )
    if add_generation_prompt:
        rendered += f"{IM_START}assistant\n"
    return rendered


def assert_glyph_template_parity(tokenizer: Any | None = None) -> None:
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USR"},
        {"role": "assistant", "content": 'CALL read_file {"id":"c1"}'},
        {"role": "tool", "content": "RESULT c1:\nstatus: success"},
    ]
    if tokenizer is None:
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        rendered = ImmutableSandboxedEnvironment().from_string(GLYPH_CHAT_TEMPLATE).render(
            messages=messages, add_generation_prompt=True
        )
    else:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    expected = render_messages(messages, add_generation_prompt=True)
    if rendered != expected:
        raise RuntimeError("GLYPH chat template and renderer differ")
