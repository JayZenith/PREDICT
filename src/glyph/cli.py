"""Command-line entry points for GLYPH data preparation and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="glyph")
    commands = parser.add_subparsers(dest="command", required=True)

    frontier = commands.add_parser("frontier", help="retain mixed-outcome pass@8 RL tasks")
    frontier.add_argument("traces", type=Path)
    frontier.add_argument("--tasks", type=Path, default=Path("data/rl_candidates.jsonl"))
    frontier.add_argument("--output", type=Path, default=Path("data/rl.jsonl"))
    frontier.add_argument("-k", type=int, default=8)

    passk = commands.add_parser("passk", help="compute task-level pass@k")
    passk.add_argument("traces", type=Path)
    passk.add_argument("-k", type=int, default=1)

    compare_cmd = commands.add_parser("compare", help="compare SFT and RLVR on identical tasks")
    compare_cmd.add_argument("sft_traces", type=Path)
    compare_cmd.add_argument("rlvr_traces", type=Path)
    compare_cmd.add_argument("-k", type=int, default=1)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "frontier":
        from .passk import select_frontier

        result = select_frontier(args.traces, args.tasks, args.output, args.k)
    elif args.command == "passk":
        from .passk import summarize

        result = summarize(args.traces, args.k)
    else:
        from .passk import compare

        result = compare(args.sft_traces, args.rlvr_traces, args.k)
    if isinstance(result, Path):
        print(result)
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
