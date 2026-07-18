"""Construct one verified semantic failure for matched recovery traces."""

from __future__ import annotations

import io
import tempfile
import tokenize
from dataclasses import dataclass
from pathlib import Path

from glyph.program import PASS, run_hidden_tests


@dataclass(frozen=True)
class Patch:
    find: str
    replace: str


@dataclass(frozen=True)
class RecoveryTrace:
    initial_code: str
    patch: Patch
    outcome: str


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
        if lines.count(line) != 1:
            continue
        replacement = _REPLACEMENTS.get(token.string)
        if replacement is None and token.type == tokenize.NUMBER and token.string.isdigit():
            replacement = str(int(token.string) + 1)
        if replacement is None or token.start[0] != token.end[0]:
            continue
        changed = line[: token.start[1]] + replacement + line[token.end[1] :]
        if changed != line and changed not in lines:
            yield _Mutation(line_index, line, changed)


def _apply(code: str, mutation: _Mutation) -> str:
    lines = code.splitlines(keepends=True)
    if lines[mutation.line] != mutation.original:
        raise ValueError("recovery mutation no longer matches its source line")
    lines[mutation.line] = mutation.changed
    return "".join(lines)


def generate_recovery(
    code: str,
    test_code: str,
    case_id: str,
    *,
    timeout: int = 5,
    gold_verified: bool = False,
) -> RecoveryTrace | None:
    with tempfile.TemporaryDirectory(prefix="predict-sft-recovery-") as temporary:
        project = Path(temporary)
        solution = project / "solution.py"
        if not gold_verified:
            solution.write_text(code, encoding="utf-8")
            gold = run_hidden_tests(project, test_code, timeout)
            if gold.outcome != PASS:
                raise RuntimeError(
                    f"gold solution failed verification for {case_id}: {gold.outcome}"
                )

        for mutation in _mutations(code):
            initial = _apply(code, mutation)
            if initial.count(mutation.changed) != 1:
                continue
            solution.write_text(initial, encoding="utf-8")
            result = run_hidden_tests(project, test_code, timeout)
            if result.outcome == PASS:
                continue
            return RecoveryTrace(
                initial_code=initial,
                patch=Patch(mutation.changed, mutation.original),
                outcome=result.outcome,
            )
    return None
