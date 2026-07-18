# /// script
# requires-python = ">=3.11"
# dependencies = ["openai==2.32.0"]
# ///
"""Sandbox-side GLYPH agent loop for Python function tasks."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import secrets
import signal
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from openai import AsyncOpenAI # async client used to call served model via OpenAI-comp endpoint


SUPPORTED_TOOLS = {"read_file", "apply_patch", "python_test"}
TOOL_NAME_RE = re.compile(r"^[A-Za-z_]\w*$")

# Parsed momdel call
@dataclass(frozen=True)
class Call:
    tool: str
    id: str # link call to its result
    params: dict[str, str]

# outcome of tool
@dataclass(frozen=True)
class Result:
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False

# turns line like CALL read_file {"id":"c1","file_path":"solution.py"} into
# a CALL object or rejects if malformed
def parse_call_line(line: str) -> tuple[Call | None, str | None]:
    stripped = line.strip()
    if not stripped.startswith("CALL "):
        return None, None
    try:
        _, rest = stripped.split(None, 1)
        tool, payload = rest.split(None, 1)
        decoded = json.loads(payload)
    except (ValueError, json.JSONDecodeError) as exc:
        return None, f"malformed CALL: {exc}"
    if not TOOL_NAME_RE.fullmatch(tool) or not isinstance(decoded, dict):
        return None, "malformed CALL"
    params = dict(decoded)
    call_id = params.pop("id", None)
    if not isinstance(call_id, str) or not call_id:
        return None, "CALL requires a string id"
    if any(not isinstance(value, str) for value in params.values()):
        return None, "CALL arguments must be strings"
    return Call(tool, call_id, params), None

# Ex model response:
# I’ll inspect the file.
# CALL read_file {"id":"c1","file_path":"solution.py"}
# CALL python_test {"id":"c2","project_path":"."}
#
# Below returns:
# [
#   Call(tool="read_file", id="c1", params={"file_path": "solution.py"}),
#   Call(tool="python_test", id="c2", params={"project_path": "."}),
# ]
# and
# []
def parse_calls(
    text: str, seen_ids: set[str] | None = None
) -> tuple[list[Call], list[str]]:
    calls: list[Call] = []
    errors: list[str] = []
    seen = set(seen_ids or ())
    for line_no, line in enumerate(text.splitlines(), 1):
        call, error = parse_call_line(line)
        if error:
            errors.append(f"line {line_no}: {error}")
        if call is None:
            continue
        if call.id in seen:
            errors.append(f"line {line_no}: duplicate CALL id {call.id}")
            continue
        seen.add(call.id)
        calls.append(call)
    return calls, errors

# Example
# root = Path("/workspace/task")
# value = "solution.py"
# returns: /workspace/task/solution.py
# to reject anything trying to access outside of root.
def confined_path(value: str, root: Path, *, require_exists: bool = False) -> Path:
    raw = Path(value)
    candidate = raw.resolve(strict=False) if raw.is_absolute() else (Path.cwd() / raw).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        candidate = (root / raw).resolve()
    candidate.relative_to(root)
    if require_exists and not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate


# Takes model's solution.py, appends hidden tests under it, runs combined file, and returns pass
# only if Python exits cleanly and reaches final success marker
# Its check.py and python reaching the marker means solution passed
def run_hidden_tests(project: Path, test_code: str, timeout: int) -> Result:
    solution = project / "solution.py"
    if not solution.is_file():
        return Result(False, "", "solution.py not found", -1)
    source = solution.read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="glyph-python-") as temporary:
        root = Path(temporary)
        check = root / "check.py"
        marker = f"GLYPH_TESTS_PASSED_{secrets.token_hex(16)}"
        check.write_text(
            f"{source.rstrip()}\n\n{test_code.rstrip()}\n\n"
            f"import sys as _glyph_sys\n_glyph_sys.__stdout__.write({marker!r} + '\\n')\n",
            encoding="utf-8",
        )
        # MBPP+ imports NumPy only for ndarray type checks and allclose. The
        # benchmark inputs are Python literals, so this tiny isolated shim is
        # sufficient and was validated against all 224 held-out gold programs.
        (root / "numpy.py").write_text(
            """class ndarray: pass
float64 = float
float32 = float
def allclose(a, b, rtol=1e-7, atol=0.0):
    if isinstance(a, (set, frozenset)) and isinstance(b, (set, frozenset)):
        a, b = sorted(a, key=repr), sorted(b, key=repr)
    seq = (list, tuple, set, frozenset)
    if isinstance(a, seq) or isinstance(b, seq):
        try:
            aa, bb = list(a), list(b)
        except TypeError:
            return False
        return len(aa) == len(bb) and all(
            allclose(x, y, rtol=rtol, atol=atol) for x, y in zip(aa, bb)
        )
    try:
        return abs(a - b) <= atol + rtol * abs(b)
    except (TypeError, ValueError):
        return a == b
""",
            encoding="utf-8",
        )
        env = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "LANG": "C.UTF-8",
            "HOME": "/tmp",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        try: # sets a minimal env and runs check.py in seperate Python process w/ timeout
            process = subprocess.Popen(
                ["python3", "-B", str(check)], # runs /tmp/.../check.py
                cwd=root,
                env=env,
                text=True,
                stdout=subprocess.PIPE, # capture stdout
                stderr=subprocess.PIPE, # captures stderr
                start_new_session=True,
            )
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
                return Result(False, "", f"hidden tests timed out after {timeout}s", -1, True)
        except OSError as exc:
            return Result(False, "", str(exc), exc.errno or -1)
    if process.returncode == 0 and stdout.strip().splitlines()[-1:] == [marker]:
        return Result(True, "hidden tests passed", "", 0)
    # Keep tests hidden: expose the error category, never assertion source or values.
    if "SyntaxError" in stderr:
        detail = "generated solution has a syntax error"
    elif "NameError" in stderr:
        detail = "generated solution raised NameError"
    elif "TypeError" in stderr:
        detail = "generated solution raised TypeError"
    elif "RecursionError" in stderr:
        detail = "generated solution raised RecursionError"
    else:
        detail = "hidden tests failed"
    return Result(False, "", detail, process.returncode)

# Receives a parsed CALL and routes it to read_file, apply_patch, or python_test
def execute_tool(call: Call, root: Path, test_code: str, timeout: int) -> Result:
    if call.tool not in SUPPORTED_TOOLS:
        return Result(False, "", f"unknown tool: {call.tool}", -1)
    try:
        if call.tool == "read_file":
            path = confined_path(call.params.get("file_path", ""), root, require_exists=True)
            return Result(True, path.read_text(encoding="utf-8")[:8000], "", 0)
        if call.tool == "apply_patch":
            path = confined_path(call.params.get("file_path", ""), root, require_exists=True)
            find = call.params.get("find")
            replace = call.params.get("replace")
            if find is None or replace is None:
                return Result(False, "", "apply_patch needs file_path, find, replace", -1)
            text = path.read_text(encoding="utf-8")
            count = text.count(find)
            if count != 1:
                return Result(False, "", f"find must occur exactly once; found {count}", -1)
            path.write_text(text.replace(find, replace, 1), encoding="utf-8")
            return Result(True, "patch applied", "", 0)
        project = confined_path(call.params.get("project_path", "."), root, require_exists=True)
        return run_hidden_tests(project, test_code, timeout)
    except (OSError, UnicodeError, ValueError) as exc:
        return Result(False, "", f"tool error: {exc}", -1)

#  converts Result into text send back to model as tool message
# Ex) RESULT c2:\nstatus: success\nstdout:\nhidden tests passed.
def result_block(call_id: str, result: Result) -> str:
    lines = [f"status: {'success' if result.success else 'failed'}"]
    if result.timed_out:
        lines.append("timed_out: true")
    if result.stdout:
        lines.append(f"stdout:\n{result.stdout.strip()}")
    if result.stderr:
        lines.append(f"stderr:\n{result.stderr.strip()}")
    return f"RESULT {call_id}:\n" + "\n".join(lines)

# cli inputs sandbox program erquires -- model endpoing, API key, model name, task folder, hidden-test path, max tool calls, per-test timeout
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--trace-prefix", required=True)
    parser.add_argument("--test-file", default=".glyph/tests.py")
    parser.add_argument("--max-tool-calls", type=int, default=8)
    parser.add_argument("--tool-timeout", type=int, default=30)
    return parser.parse_args()

# parses CLI args, resolves task root, loads initial chat messgaes and hidden tests, creates async OpenAI client, enters agnet loop
async def main() -> None:
    args = parse_args()
    # --trace-prefix value is used to turn it into abs path under current sandbox dir, rejects paths
    # outside sandbox, and requires folder to exist
    root = confined_path(args.trace_prefix, Path.cwd().resolve(), require_exists=True)
    # loads starting system/user messages
    messages = json.loads(os.environ["GLYPH_INITIAL_MESSAGES"])
    # load hidden tests
    test_code = Path(args.test_file).read_text(encoding="utf-8")
    # connect to served model
    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=1800.0,
        max_retries=0,
    )
    executed: list[Call] = []
    results: dict[str, Result] = {}
    errors: list[str] = []
    try:
        while True: # loop stops when odel gives no tool call, makes protocol error, or tool-call limit
            # repeaddtly ask model for next response
            completion = await client.chat.completions.create(model=args.model, messages=messages)
            assistant = completion.choices[0].message.content or ""
            # store the response
            messages.append({"role": "assistant", "content": assistant})
            # extract any CALL ... lines
            calls, parse_errors = parse_calls(assistant, {call.id for call in executed})
            errors.extend(parse_errors)
            if parse_errors or not calls:
                break
            remaining = args.max_tool_calls - len(executed)
            if remaining <= 0:
                break
            for call in calls[:remaining]:
                result = execute_tool(call, root, test_code, args.tool_timeout)
                executed.append(call)
                results[call.id] = result
                messages.append( # feed results of tools back to model
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result_block(call.id, result),
                    }
                )
    finally:
        await client.close()
    record = {
        "calls": [asdict(call) for call in executed],
        "results": {call_id: asdict(result) for call_id, result in results.items()},
        "protocol_errors": errors,
    }
    path = Path(".glyph/trace.json") # writes every call, result, and protocol error here
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
