from glyph.chat import (
    DEFAULT_SYSTEM_PROMPT,
    IM_END,
    IM_START,
    assert_glyph_template_parity,
    render_messages,
)


def test_render_messages_preserves_agent_roles_and_chatml_markers() -> None:
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": 'CALL read_file {"id":"c1"}'},
        {"role": "tool", "content": "RESULT c1:\nstatus: success"},
        {"role": "assistant", "content": "FINAL: done"},
    ]
    rendered = render_messages(messages, add_generation_prompt=True)
    for role in ("system", "user", "assistant", "tool"):
        assert f"{IM_START}{role}\n" in rendered
    assert rendered.count(IM_START) == 6
    assert rendered.count(IM_END) == 5
    assert "CALL read_file" in rendered
    assert "RESULT c1:" in rendered
    assert "FINAL: done" in rendered
    assert rendered.endswith(f"{IM_START}assistant\n")


def test_system_prompt_does_not_hint_the_tool_protocol() -> None:
    assert "CALL" not in DEFAULT_SYSTEM_PROMPT
    assert "read_file" not in DEFAULT_SYSTEM_PROMPT
    assert "python_test" not in DEFAULT_SYSTEM_PROMPT


def test_python_and_jinja_chatml_renderers_match() -> None:
    assert_glyph_template_parity()
