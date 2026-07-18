"""Verifiers v1 harness for the GLYPH CALL/RESULT/FINAL agent loop."""

from __future__ import annotations

import json
from pathlib import Path

import verifiers.v1 as vf


PROGRAM_SOURCE = (Path(__file__).resolve().parent / "program.py").read_text()

# Define Sandbox defaults: Python image, /worksapce, network on, CPUP/RAM/disk, max tool calls, timeout
class GlyphHarnessConfig(vf.HarnessConfig):
    runtime: vf.RuntimeConfig = vf.PrimeConfig(
        image="python:3.12-slim-bookworm",
        workdir="/workspace",
        network_access=True,
        labels=["glyph", "verifiers-v1", "python", "mbpp"],
        cpu=1.0,
        memory=2.0,
        disk=2.0,
        idle_timeout=600,
    )
    max_tool_calls: int = 8
    tool_timeout: int = 30


class GlyphHarness(vf.Harness[GlyphHarnessConfig]):
    SUPPORTS_MESSAGE_PROMPT = True

    # preps program.py as a runnable uv script inside runtime
    async def setup(self, runtime: vf.Runtime) -> None:
        await runtime.prepare_uv_script(PROGRAM_SOURCE, self.config.resolved_env)

    # Prime-RL reaches through Verifiers
    async def launch(
        self,
        ctx: vf.ModelContext,
        trace: vf.Trace,
        runtime: vf.Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> vf.ProgramResult:
        if mcp_urls:
            raise ValueError("GLYPH does not use MCP tools")
        # get task's system/user messages
        _, prompt = self.resolve_prompt(trace.task.data)
        if not isinstance(prompt, list):
            raise ValueError("GLYPH tasks require structured system/user messages")
        messages = [
            message.model_dump(mode="json") if hasattr(message, "model_dump") else dict(message)
            for message in prompt
        ]
        data = trace.task.data
        # Materializes program.py in sandbox
        program = await runtime.prepare_uv_script(PROGRAM_SOURCE, self.config.resolved_env)
        # Sandbox script will know which served model endpoing to call and which task folder/tests to use
        argv = [
            *program,
            f"--base-url={endpoint}",
            f"--api-key={secret}",
            f"--model={ctx.model}",
            f"--trace-prefix={data.trace_prefix}",
            "--test-file=.glyph/tests.py",
            f"--max-tool-calls={self.config.max_tool_calls}",
            f"--tool-timeout={self.config.tool_timeout}",
        ]
        env = {
            **self.config.resolved_env,
            "GLYPH_INITIAL_MESSAGES": json.dumps(messages),
        }
        # Run program.py in sandbox
        return await runtime.run_program(argv, env)


__all__ = ["GlyphHarness"]
