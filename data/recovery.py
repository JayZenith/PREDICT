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
class TwoStepRecoveryTrace:
    """Two independently-verified failing mutations of the same gold code,
    fixed sequentially: blank -> first_code (fails) -> second_code (fails,
    an unrelated wrong guess) -> gold (passes)."""

    first_code: str
    first_outcome: str
    second_patch: Patch
    second_code: str
    second_outcome: str
    final_patch: Patch


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


def generate_two_step_recovery(
    code: str,
    test_code: str,
    case_id: str,
    *,
    timeout: int = 5,
    gold_verified: bool = False,
) -> TwoStepRecoveryTrace | None:
    """Find two distinct mutations that each independently fail alone against
    gold, then chain them: first_code (mutation A) -> second_code (mutation
    B, an unrelated wrong guess replacing A) -> gold. The middle patch reverts
    A and applies B in one block-spanning find/replace; the final patch
    reverts B alone, landing on the exact original gold text."""
    with tempfile.TemporaryDirectory(prefix="predict-sft-recovery2-") as temporary:
        project = Path(temporary)
        solution = project / "solution.py"
        if not gold_verified:
            solution.write_text(code, encoding="utf-8")
            gold = run_hidden_tests(project, test_code, timeout)
            if gold.outcome != PASS:
                raise RuntimeError(
                    f"gold solution failed verification for {case_id}: {gold.outcome}"
                )

        lines = code.splitlines(keepends=True)
        failing: list[tuple[_Mutation, str]] = []
        for mutation in _mutations(code):
            candidate = _apply(code, mutation)
            if candidate.count(mutation.changed) != 1:
                continue
            solution.write_text(candidate, encoding="utf-8")
            result = run_hidden_tests(project, test_code, timeout)
            if result.outcome == PASS:
                continue
            failing.append((mutation, result.outcome))
            if len(failing) < 2:
                continue
            mutation_a, outcome_a = failing[0]
            for mutation_b, outcome_b in failing[1:]:
                lo, hi = sorted((mutation_a.line, mutation_b.line))
                first_code = _apply(code, mutation_a)
                first_lines = first_code.splitlines(keepends=True)
                second_lines = list(lines)
                second_lines[mutation_b.line] = mutation_b.changed
                second_code = "".join(second_lines)

                first_block = "".join(first_lines[lo : hi + 1])
                second_block = "".join(second_lines[lo : hi + 1])
                if first_block == second_block or first_code.count(first_block) != 1:
                    continue
                if second_code.count(mutation_b.changed) != 1:
                    continue

                return TwoStepRecoveryTrace(
                    first_code=first_code,
                    first_outcome=outcome_a,
                    second_patch=Patch(first_block, second_block),
                    second_code=second_code,
                    second_outcome=outcome_b,
                    final_patch=Patch(mutation_b.changed, mutation_b.original),
                )
    return None
