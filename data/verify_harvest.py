"""Execute every harvested SFT trace and check its declared outcomes.

`data.validate` performs this check for the prepared dataset as a whole. The
harvested set is a standalone pair of SFT files, so this runs the same
per-trace verification against it: every candidate is re-executed and the
predicted, verified, decision and visible-test sequences must match what the
trace shape promises.

    uv run python -m data.verify_harvest --sft data/sft_harvested
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

from .prepare import PLACEHOLDER
from .validate import _rows, _verify_trace, validate_sft


def verify(sft_dir: Path, tasks_path: Path, *, timeout: int = 5) -> dict:
    tasks = {row["case_id"]: row for row in _rows(tasks_path)}
    report: dict = {}
    for arm in ("a", "b"):
        path = sft_dir / f"arm_{arm}" / "train.jsonl"
        rows = _rows(path)
        outcomes: Counter[str] = Counter()
        with tempfile.TemporaryDirectory() as workdir:
            project = Path(workdir) / "project"
            project.mkdir()
            for row in rows:
                task = tasks.get(row["case_id"])
                if task is None:
                    raise ValueError(f"{row['case_id']} is not in {tasks_path}")
                (project / "solution.py").write_text(PLACEHOLDER, encoding="utf-8")
                outcomes += _verify_trace(row, task, project, timeout)
        report[f"arm_{arm}"] = {
            "traces": len(rows),
            "verified_outcomes": dict(outcomes),
            **validate_sft(path),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft", type=Path, default=Path("data/sft_harvested"))
    parser.add_argument("--tasks", type=Path, default=Path("data/arm_b_train.jsonl"))
    parser.add_argument("--timeout", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(verify(args.sft, args.tasks, timeout=args.timeout), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
