"""Reject SFT traces that would be truncated or teach an incomplete ending."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from glyph.chat import render_messages


DEFAULT_MODEL = "Qwen/Qwen3-4B-Base"
SFT_MAX_TOKENS = 1024


def validate_sft(
    path: Path,
    *,
    max_tokens: int = SFT_MAX_TOKENS,
    model: str = DEFAULT_MODEL,
    tokenizer: Any | None = None,
) -> dict[str, int | str]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model)

    longest = 0
    longest_case = ""
    count = 0
    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            row = json.loads(line)
            messages = row.get("messages") or []
            case_id = str(row.get("case_id", line_no))
            if (
                not messages
                or messages[-1].get("role") != "assistant"
                or not str(messages[-1].get("content", "")).strip().startswith("FINAL:")
            ):
                raise ValueError(f"{path}:{line_no} ({case_id}) lacks a terminal FINAL:")
            tokens = len(
                tokenizer.encode(
                    render_messages(messages),
                    add_special_tokens=False,
                )
            )
            if tokens > max_tokens:
                raise ValueError(
                    f"{path}:{line_no} ({case_id}) has {tokens} tokens; "
                    f"limit is {max_tokens}"
                )
            if tokens > longest:
                longest = tokens
                longest_case = case_id
            count += 1
    if not count:
        raise ValueError(f"{path} contains no SFT traces")
    return {"traces": count, "max_tokens": longest, "longest_case": longest_case}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, nargs="?", default=Path("data/sft.jsonl"))
    parser.add_argument("--max-tokens", type=int, default=SFT_MAX_TOKENS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    print(json.dumps(validate_sft(args.path, max_tokens=args.max_tokens, model=args.model)))


if __name__ == "__main__":
    main()
