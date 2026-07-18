import json
from pathlib import Path

import pytest
import verifiers.v1 as vf
from verifiers.v1.runtimes import make_runtime

from glyph.harness import GlyphHarnessConfig
from glyph.taskset import GlyphTaskset, GlyphTasksetConfig, load_rows


def _taskset(tmp_path: Path) -> GlyphTaskset:
    project = tmp_path / "blueprints/case"
    project.mkdir(parents=True)
    (project / "solution.py").write_text("# Write your function here.\n")
    data = tmp_path / "tasks.jsonl"
    data.write_text(
        json.dumps(
            {
                "case_id": "mbpp_601",
                "source": "mbpp",
                "task_id": 601,
                "prompt": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "implement"},
                ],
                "blueprint_root": "blueprints/case",
                "trace_prefix": "data/blueprints/case",
                "test_code": "assert answer() == 42\n",
            }
        )
        + "\n"
    )
    return GlyphTaskset(GlyphTasksetConfig(id="glyph", data_path=str(data)))


def _trace(task, turns: list[tuple[str, str, str | None]]) -> vf.Trace:
    trace = vf.Trace(
        task=vf.TraceTask(type="GlyphTask", data=task.data),
        state=vf.State(),
    )
    parent = None
    for message in [
        vf.SystemMessage(content="system"),
        vf.UserMessage(content="implement"),
    ]:
        trace.nodes.append(vf.MessageNode(parent=parent, message=message))
        parent = len(trace.nodes) - 1
    for role, content, call_id in turns:
        if role == "assistant":
            message = vf.AssistantMessage(content=content)
            sampled = True
        else:
            message = vf.ToolMessage(content=content, tool_call_id=call_id or "")
            sampled = False
        trace.nodes.append(vf.MessageNode(parent=parent, message=message, sampled=sampled))
        parent = len(trace.nodes) - 1
    return trace


def _runtime_record(*, errors: list[str] | None = None) -> dict:
    return {
        "calls": [{"tool": "python_test", "id": "c1", "params": {}}],
        "results": {"c1": {"success": True}},
        "protocol_errors": errors or [],
    }


@pytest.mark.asyncio
async def test_task_setup_and_binary_reward(tmp_path: Path) -> None:
    [task] = _taskset(tmp_path).load()
    trace = vf.Trace(task=vf.TraceTask(type="GlyphTask", data=task.data))
    runtime = make_runtime(vf.SubprocessConfig(), name="glyph-python-test")
    await runtime.start()
    try:
        await task.setup(trace, runtime)
        assert b"Write your function" in await runtime.read(
            "data/blueprints/case/solution.py"
        )
        assert b"answer() == 42" in await runtime.read(".glyph/tests.py")
    finally:
        await runtime.stop()

    trace = _trace(
        task,
        [
            ("assistant", 'CALL python_test {"id":"c1","project_path":"."}', None),
            ("tool", "RESULT c1:\nstatus: success", "c1"),
            ("assistant", "FINAL: implemented and tested.", None),
        ],
    )
    trace.info["glyph"] = _runtime_record()
    assert await task.mbpp_reward(trace) == 1.0
    assert await task.passed(trace) == 1.0

    # Evaluation is computed once for both reward and metric.
    assert trace.info["glyph_evaluation"] == (1.0, True)
    trace.info["glyph"]["results"]["c1"]["success"] = False
    assert await task.passed(trace) == 1.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("turns", "errors"),
    [
        (
            [
                ("assistant", 'CALL python_test {"id":"c1"}', None),
                ("tool", "RESULT c1:\nstatus: success", "c1"),
                ("assistant", "Tests pass, but this is not FINAL syntax.", None),
            ],
            [],
        ),
        (
            [
                (
                    "assistant",
                    'CALL python_test {"id":"c1"}\nRESULT c1:\nstatus: success',
                    None,
                ),
                ("assistant", "FINAL: claimed success.", None),
            ],
            [],
        ),
        (
            [
                ("assistant", 'CALL python_test {"id":"c1"}', None),
                ("tool", "RESULT other:\nstatus: success", "other"),
                ("assistant", "FINAL: claimed success.", None),
            ],
            [],
        ),
        (
            [
                ("assistant", 'CALL python_test {"id":"c1"}', None),
                ("tool", "RESULT c1:\nstatus: success", "c1"),
                ("assistant", "FINAL: claimed success.", None),
            ],
            ["line 1: malformed CALL"],
        ),
    ],
)
async def test_runtime_success_cannot_bypass_clean_trace_gate(
    tmp_path: Path,
    turns: list[tuple[str, str, str | None]],
    errors: list[str],
) -> None:
    [task] = _taskset(tmp_path).load()
    trace = _trace(task, turns)
    trace.info["glyph"] = _runtime_record(errors=errors)
    assert await task.mbpp_reward(trace) == 0.0
    assert await task.passed(trace) == 0.0


@pytest.mark.asyncio
async def test_later_tool_turn_is_not_a_separate_penalty(tmp_path: Path) -> None:
    [task] = _taskset(tmp_path).load()
    trace = _trace(
        task,
        [
            ("assistant", 'CALL python_test {"id":"c1"}', None),
            ("tool", "RESULT c1:\nstatus: success", "c1"),
            ("assistant", 'CALL apply_patch {"id":"c2"}', None),
            ("tool", "RESULT c2:\nstatus: success", "c2"),
            ("assistant", "FINAL: done.", None),
        ],
    )
    trace.info["glyph"] = {
        "calls": [
            {"tool": "python_test", "id": "c1", "params": {}},
            {"tool": "apply_patch", "id": "c2", "params": {}},
        ],
        "results": {"c1": {"success": True}, "c2": {"success": True}},
        "protocol_errors": [],
    }
    assert await task.mbpp_reward(trace) == 1.0


@pytest.mark.asyncio
async def test_truncated_rl_trace_hard_fails_before_scoring(tmp_path: Path) -> None:
    [task] = _taskset(tmp_path).load()
    trace = _trace(task, [("assistant", "FINAL: cut off", None)])
    trace.stop("max_total_tokens")
    with pytest.raises(vf.TaskError, match="truncated rollout"):
        await task.finalize(trace, None)
    assert await task.mbpp_reward(trace) == 0.0

    overlong = _trace(task, [("assistant", "FINAL: too long", None)])
    overlong.nodes[-1].token_ids = [1] * (task.config.max_trace_tokens + 1)
    overlong.nodes[-1].mask = [True] * (task.config.max_trace_tokens + 1)
    with pytest.raises(vf.TaskError, match="exceeding the training limit"):
        await task.finalize(overlong, None)


def test_taskset_metadata_and_runtime_defaults(tmp_path: Path) -> None:
    [task] = _taskset(tmp_path).load()
    assert task.data.source_task_id == 601
    assert task.data.test_code == "assert answer() == 42\n"
    config = GlyphHarnessConfig(id="glyph")
    assert config.runtime.image == "python:3.12-slim-bookworm"
    assert config.max_tool_calls == 8


def test_taskset_requires_data_and_reports_bad_json(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="explicit data_path"):
        GlyphTaskset(GlyphTasksetConfig()).load()
    path = tmp_path / "bad.jsonl"
    path.write_text("not-json\n")
    with pytest.raises(ValueError, match=r"bad\.jsonl:1"):
        load_rows(str(path))
