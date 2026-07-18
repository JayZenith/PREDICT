"""Load Python function tasks and score real hidden-test execution."""

from __future__ import annotations

import asyncio
import io
import json
import tarfile
from pathlib import Path

import verifiers.v1 as vf

from .chat import message_content, message_role, message_tool_call_id


def load_rows(data_path: str, max_samples: int | None = None) -> list[dict]:
    path = Path(data_path)
    rows: list[dict] = []
    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if max_samples is not None and len(rows) >= max_samples:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
            required = {"case_id", "prompt", "blueprint_root", "test_code"}
            if not isinstance(row, dict) or not required <= row.keys():
                raise ValueError(f"{path}:{line_no}: missing required task fields")
            if not isinstance(row["prompt"], list):
                raise ValueError(f"{path}:{line_no}: prompt must be a message list")
            rows.append(row)
    return rows


class GlyphTaskData(vf.TaskData):
    case_id: str
    source: str = "mbpp"
    source_task_id: int
    blueprint_root: str
    trace_prefix: str
    test_code: str


class GlyphTaskConfig(vf.TaskConfig):
    max_trace_tokens: int = 4096


def _archive_blueprint(source: Path, trace_prefix: str) -> bytes:
    source = source.resolve(strict=True)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        archive.add(source, arcname=trace_prefix, recursive=True)
    return buffer.getvalue()


class GlyphTask(vf.Task[GlyphTaskData, vf.State, GlyphTaskConfig]):
    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        payload = await asyncio.to_thread(
            _archive_blueprint, Path(self.data.blueprint_root), self.data.trace_prefix
        )
        await runtime.write(".glyph/task.tar.gz", payload)
        await runtime.write(".glyph/tests.py", self.data.test_code.encode("utf-8"))
        result = await runtime.run(
            ["tar", "-xzf", ".glyph/task.tar.gz", "--no-same-owner", "--no-same-permissions"],
            {},
        )
        if result.exit_code != 0:
            raise vf.SandboxError(f"could not materialize Python task: {result.stderr.strip()}")

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        if reason := self._truncation_reason(trace):
            raise vf.TaskError(reason)
        payload = await runtime.read(".glyph/trace.json")
        trace.info["glyph"] = json.loads(payload.decode("utf-8"))

    def _truncation_reason(self, trace: vf.Trace) -> str | None:
        if trace.is_truncated:
            return f"truncated rollout: {trace.stop_condition or 'generation length'}"
        longest = max((branch.num_total_tokens for branch in trace.branches), default=0)
        if longest > self.config.max_trace_tokens:
            trace.stop_condition = "max_total_tokens"
            return (
                f"rollout has {longest} tokens, exceeding the training limit "
                f"of {self.config.max_trace_tokens}"
            )
        return None

    def _evaluate(self, trace: vf.Trace) -> tuple[float, bool]:
        cached = trace.info.get("glyph_evaluation")
        if cached is not None:
            return float(cached[0]), bool(cached[1])

        if self._truncation_reason(trace):
            evaluation = (0.0, False)
            trace.info["glyph_evaluation"] = evaluation
            return evaluation

        state = trace.info.get("glyph") or {}
        calls = state.get("calls") or []
        results = state.get("results") or {}
        successful_call_id = next(
            (
                call.get("id")
                for call in calls
                if call.get("tool") == "python_test"
                and bool((results.get(call.get("id")) or {}).get("success"))
            ),
            None,
        )
        messages = trace.branches[-1].messages if trace.branches else []
        valid = bool(
            successful_call_id
            and not state.get("protocol_errors")
            and messages
            and message_role(messages[-1]) == "assistant"
            and message_content(messages[-1]).strip().startswith("FINAL:")
            and any(
                message_role(message) == "tool"
                and message_tool_call_id(message) == successful_call_id
                for message in messages[:-1]
            )
        )
        evaluation = (float(valid), valid)
        trace.info["glyph_evaluation"] = evaluation
        return evaluation

    @vf.reward(weight=1.0)
    async def mbpp_reward(self, trace: vf.Trace) -> float:
        return self._evaluate(trace)[0]

    @vf.metric
    async def passed(self, trace: vf.Trace) -> float:
        return float(self._evaluate(trace)[1])


class GlyphTasksetConfig(vf.TasksetConfig):
    data_path: str | None = None
    max_samples: int | None = None
    task: GlyphTaskConfig = GlyphTaskConfig()


def _blueprint_path(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve(strict=True) if path.is_absolute() else (root / path).resolve(strict=True)


class GlyphTaskset(vf.Taskset[GlyphTask, GlyphTasksetConfig]):
    def load(self) -> list[GlyphTask]:
        if not self.config.data_path:
            raise ValueError("GlyphTaskset requires an explicit data_path")
        data_path = Path(self.config.data_path).expanduser().resolve(strict=True)
        rows = load_rows(str(data_path), self.config.max_samples)
        tasks: list[GlyphTask] = []
        for idx, row in enumerate(rows):
            tasks.append(
                GlyphTask(
                    GlyphTaskData(
                        idx=idx,
                        name=row["case_id"],
                        prompt=row["prompt"],
                        case_id=row["case_id"],
                        source=row.get("source", "mbpp"),
                        source_task_id=int(row.get("task_id", idx)),
                        blueprint_root=str(_blueprint_path(row["blueprint_root"], data_path.parent)),
                        trace_prefix=row.get("trace_prefix") or row["blueprint_root"],
                        test_code=row["test_code"],
                    ),
                    self.config.task,
                )
            )
        return tasks


__all__ = ["GlyphTaskset"]
