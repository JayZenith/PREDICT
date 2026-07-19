import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from glyph.program import (
    ASSERTION_FAILURE,
    PASS,
    Call,
    execute_tool,
    parse_calls,
    parse_prediction_decision,
    validate_turn_shape,
)


def test_protocol_parses_calls_and_rejects_duplicate_ids() -> None:
    calls, errors = parse_calls(
        'CALL read_file {"id":"c1","file_path":"solution.py"}\n'
        'CALL python_test {"id":"c1","project_path":"."}'
    )
    assert [call.tool for call in calls] == ["read_file"]
    assert errors == ["line 2: duplicate CALL id c1"]

    calls, errors = parse_calls(
        'CALL python_test {"id":"c1","project_path":"."}', {"c1"}
    )
    assert calls == []
    assert errors == ["line 1: duplicate CALL id c1"]


def test_arm_b_prediction_and_decision_are_strict() -> None:
    assert parse_prediction_decision(
        "<PREDICTION>PASS</PREDICTION>\n<DECISION>KEEP</DECISION>"
    ) == ("PASS", "KEEP", [])
    prediction, decision, errors = parse_prediction_decision(
        "<PREDICTION>MAYBE</PREDICTION>\n<DECISION>WAIT</DECISION>"
    )
    assert prediction == "MAYBE"
    assert decision == "WAIT"
    assert errors == [
        "unknown prediction outcome: MAYBE",
        "unknown decision: WAIT",
    ]


def test_runtime_turn_shape_matches_sft_turns() -> None:
    [call], errors = parse_calls(
        'CALL read_file {"id":"c1","file_path":"solution.py"}'
    )
    assert not errors
    assert not validate_turn_shape(
        'CALL read_file {"id":"c1","file_path":"solution.py"}',
        [call],
        arm="a",
        pending_candidate=False,
    )
    assert validate_turn_shape(
        'CALL read_file {"id":"c1","file_path":"solution.py"}\njunk',
        [call],
        arm="a",
        pending_candidate=False,
    ) == ["CALL turn cannot contain additional text"]
    assert not validate_turn_shape(
        (
            "<PREDICTION>PASS</PREDICTION>\n"
            "<DECISION>KEEP</DECISION>\n"
            'CALL python_test {"id":"c2","project_path":"."}'
        ),
        [Call("python_test", "c2", {"project_path": "."})],
        arm="b",
        pending_candidate=True,
    )
    assert not validate_turn_shape(
        "FINAL: done", [], arm="a", pending_candidate=False
    )
    assert validate_turn_shape(
        "FINAL: done\nextra", [], arm="a", pending_candidate=False
    )


def test_python_agent_tools_patch_and_run_hidden_tests(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    source = project / "solution.py"
    source.write_text("# Write your function here.\n")

    read = execute_tool(
        Call("read_file", "c1", {"file_path": str(source)}), project, "", 2
    )
    assert read.success and "Write your function" in read.stdout

    patch = execute_tool(
        Call(
            "apply_patch",
            "c2",
            {
                "file_path": str(source),
                "find": "# Write your function here.\n",
                "replace": "def add(a, b):\n    return a + b\n",
            },
        ),
        project,
        "",
        2,
    )
    assert patch.success

    passed = execute_tool(
        Call("python_test", "c3", {"project_path": str(project)}),
        project,
        "assert add(2, 3) == 5\n",
        2,
    )
    assert passed.success and passed.stdout == "hidden tests passed"


def test_failed_tests_do_not_reveal_hidden_assertions(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "solution.py").write_text("def answer():\n    return 0\n")
    result = execute_tool(
        Call("python_test", "c1", {"project_path": str(project)}),
        project,
        "assert answer() == 42\n",
        2,
    )
    assert not result.success
    assert result.stderr == "hidden tests failed"
    assert "42" not in result.stderr


def test_clean_exit_before_tests_is_not_a_pass(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "solution.py").write_text("raise SystemExit(0)\n")
    result = execute_tool(
        Call("python_test", "c1", {"project_path": str(project)}),
        project,
        "assert False\n",
        2,
    )
    assert not result.success


def test_full_agent_loop_records_read_patch_test_and_final(tmp_path: Path) -> None:
    project = tmp_path / "data/project"
    project.mkdir(parents=True)
    (project / "solution.py").write_text("# Write your function here.\n")

    requests = 0

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            nonlocal requests
            requests += 1
            length = int(self.headers["Content-Length"])
            body = json.loads(self.rfile.read(length))
            assert body["stop_token_ids"] == [151645]
            assert body["stop"] == "\n"
            messages = body["messages"]
            last = messages[-1]
            if last["role"] == "user":
                content = 'CALL read_file {"id":"c1","file_path":"data/project/solution.py"}'
            elif "RESULT c1:" in last["content"]:
                content = (
                    'CALL apply_patch {"id":"c2","file_path":"data/project/solution.py",'
                    '"find":"# Write your function here.\\n",'
                    '"replace":"def answer():\\n    return 42\\n"}'
                )
            elif "RESULT c2:" in last["content"]:
                content = 'CALL python_test {"id":"c3","project_path":"data/project"}'
            else:
                content = "FINAL: implemented and tested the function."
            body = json.dumps(
                {
                    "id": "mock",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "mock",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    hidden = tmp_path / ".glyph/tests.py"
    hidden.parent.mkdir()
    hidden.write_text("assert answer() == 42\n")
    env = {
        **os.environ,
        "GLYPH_INITIAL_MESSAGES": json.dumps(
            [{"role": "system", "content": "system"}, {"role": "user", "content": "task"}]
        ),
    }
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "glyph.program",
                f"--base-url=http://127.0.0.1:{server.server_port}/v1",
                "--api-key=test",
                "--model=mock",
                "--trace-prefix=data/project",
                "--max-tool-calls=3",
                "--tool-timeout=2",
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )
    finally:
        server.shutdown()
        thread.join()
    assert result.returncode == 0, result.stderr
    record = json.loads((tmp_path / ".glyph/trace.json").read_text())
    assert [call["tool"] for call in record["calls"]] == [
        "read_file",
        "apply_patch",
        "python_test",
    ]
    assert record["results"]["c3"]["success"] is True
    assert record["final_verification"]["success"] is True
    assert "return 42" in (project / "solution.py").read_text()
    assert requests == 4


def test_arm_b_shadow_tests_rejected_candidate_without_exposing_result(
    tmp_path: Path,
) -> None:
    project = tmp_path / "data/project"
    project.mkdir(parents=True)
    (project / "solution.py").write_text("# Write your function here.\n")

    responses = iter(
        [
            'CALL read_file {"id":"c1","file_path":"data/project/solution.py"}',
            (
                'CALL apply_patch {"id":"c2","file_path":"data/project/solution.py",'
                '"find":"# Write your function here.\\n",'
                '"replace":"def answer():\\n    return 0\\n"}'
            ),
            (
                "<PREDICTION>ASSERTION_FAILURE</PREDICTION>\n"
                "<DECISION>REVISE</DECISION>\n"
                'CALL apply_patch {"id":"c3","file_path":"data/project/solution.py",'
                '"find":"    return 0\\n","replace":"    return 42\\n"}'
            ),
            (
                "<PREDICTION>PASS</PREDICTION>\n"
                "<DECISION>KEEP</DECISION>\n"
                'CALL python_test {"id":"c4","project_path":"data/project"}'
            ),
            "FINAL: implemented and tested the function.",
        ]
    )
    visible_requests: list[list[dict]] = []
    stop_sequences: list[str | None] = []
    include_stop: list[bool | None] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers["Content-Length"])
            request = json.loads(self.rfile.read(length))
            visible_requests.append(request["messages"])
            stop_sequences.append(request.get("stop"))
            include_stop.append(request.get("include_stop_str_in_output"))
            content = next(responses)
            body = json.dumps(
                {
                    "id": "mock",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "mock",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    hidden = tmp_path / ".glyph/tests.py"
    hidden.parent.mkdir()
    hidden.write_text("assert answer() == 42\n")
    env = {
        **os.environ,
        "GLYPH_INITIAL_MESSAGES": json.dumps(
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "task"},
            ]
        ),
    }
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "glyph.program",
                f"--base-url=http://127.0.0.1:{server.server_port}/v1",
                "--api-key=test",
                "--model=mock",
                "--trace-prefix=data/project",
                "--arm=b",
                "--max-tool-calls=4",
                "--tool-timeout=2",
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )
    finally:
        server.shutdown()
        thread.join()
    assert result.returncode == 0, result.stderr
    record = json.loads((tmp_path / ".glyph/trace.json").read_text())
    assert [call["tool"] for call in record["calls"]] == [
        "read_file",
        "apply_patch",
        "apply_patch",
        "python_test",
    ]
    [rejected, kept] = record["prediction_targets"]
    assert (rejected["actual"], rejected["shadow"]) == (
        ASSERTION_FAILURE,
        True,
    )
    assert (kept["actual"], kept["shadow"]) == (PASS, False)
    assert record["final_verification"]["outcome"] == PASS
    assert stop_sequences == ["\n", "\n", "}\n", "}\n", "\n"]
    assert include_stop == [None, None, True, True, None]
    assert all(
        "hidden tests failed"
        not in "\n".join(message.get("content", "") for message in messages)
        for messages in visible_requests
    )
