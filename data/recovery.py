"""Generate verified one- and two-phase SFT recovery trajectories."""

from __future__ import annotations

import io
import tempfile
import tokenize
from dataclasses import dataclass
from pathlib import Path

from glyph.program import run_hidden_tests


@dataclass(frozen=True)
class Patch:
    find: str
    replace: str


@dataclass(frozen=True)
class RecoveryTrace:
    initial_code: str
    patches: tuple[Patch, ...]


@dataclass(frozen=True)
class _Mutation:
    line: int
    original: str
    changed: str


_REPLACEMENTS = {
    "<=": "<",
    ">=": ">",
    "==": "!=",
    "!=": "==",
    "+": "-",
    "-": "+",
    "*": "+",
    "and": "or",
    "or": "and",
    "min": "max",
    "max": "min",
    "True": "False",
    "False": "True",
}


def _mutations(code: str):
    lines = code.splitlines(keepends=True)
    for token in tokenize.generate_tokens(io.StringIO(code).readline):
        line_index = token.start[0] - 1
        if line_index >= len(lines):
            continue
        line = lines[line_index]
        if line.lstrip().startswith(("def ", "import ", "from ")):
            continue
        replacement = _REPLACEMENTS.get(token.string)
        if replacement is None and token.type == tokenize.NUMBER and token.string.isdigit():
            replacement = str(int(token.string) + 1)
        if replacement is None or token.start[0] != token.end[0]:
            continue
        changed = line[: token.start[1]] + replacement + line[token.end[1] :]
        if changed != line and changed not in lines:
            yield _Mutation(line_index, line, changed)


def _apply(code: str, mutations: tuple[_Mutation, ...]) -> str:
    lines = code.splitlines(keepends=True)
    for mutation in mutations:
        if lines[mutation.line] != mutation.original:
            raise ValueError("recovery mutations overlap")
        lines[mutation.line] = mutation.changed
    return "".join(lines)


def generate_recovery(
    code: str,
    test_code: str,
    case_id: str,
    *,
    phases: int,
    timeout: int = 5,
) -> RecoveryTrace | None:
    if phases not in (1, 2):
        raise ValueError("recovery phases must be 1 or 2")
    with tempfile.TemporaryDirectory(prefix="glyph-sft-recovery-") as temporary:
        project = Path(temporary)
        solution = project / "solution.py"
        solution.write_text(code, encoding="utf-8")
        if not run_hidden_tests(project, test_code, timeout).success:
            raise RuntimeError(f"gold solution failed verification for {case_id}")

        failing: list[_Mutation] = []
        for mutation in _mutations(code):
            solution.write_text(_apply(code, (mutation,)), encoding="utf-8")
            if run_hidden_tests(project, test_code, timeout).success:
                continue
            if phases == 1:
                return RecoveryTrace(
                    initial_code=_apply(code, (mutation,)),
                    patches=(Patch(mutation.changed, mutation.original),),
                )
            for earlier in failing:
                if earlier.line == mutation.line or earlier.changed == mutation.changed:
                    continue
                pair = (earlier, mutation)
                initial = _apply(code, pair)
                solution.write_text(initial, encoding="utf-8")
                if not run_hidden_tests(project, test_code, timeout).success:
                    return RecoveryTrace(
                        initial_code=initial,
                        patches=(
                            Patch(earlier.changed, earlier.original),
                            Patch(mutation.changed, mutation.original),
                        ),
                    )
            failing.append(mutation)
    return None
