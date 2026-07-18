"""Command-line reporting for PREDICT evaluation traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="glyph")
    commands = parser.add_subparsers(dest="command", required=True)

    passk = commands.add_parser("passk", help="compute task-level pass@k")
    passk.add_argument("traces", type=Path)
    passk.add_argument("-k", type=int, default=1)

    report = commands.add_parser(
        "report", help="report PREDICT pass@1, recovery, prediction, and efficiency metrics"
    )
    report.add_argument("traces", type=Path)

    compare_cmd = commands.add_parser("compare", help="compare SFT and RLVR on identical tasks")
    compare_cmd.add_argument("sft_traces", type=Path)
    compare_cmd.add_argument("rlvr_traces", type=Path)
    compare_cmd.add_argument("-k", type=int, default=1)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "passk":
        from .passk import summarize

        result = summarize(args.traces, args.k)
    elif args.command == "report":
        from .passk import report

        result = report(args.traces)
    else:
        from .passk import compare

        result = compare(args.sft_traces, args.rlvr_traces, args.k)
    if isinstance(result, Path):
        print(result)
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
