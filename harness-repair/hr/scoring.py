"""Parsing and scoring. A call is *unparsed* if no answer can be read from it (scored wrong),
*truncated* if it stopped at the token limit, *non-committed* if it stopped by itself unparsed.

Only the answer region is read: with thinking on, the text after the last '</think>'; a call that
never closed its thinking has no answer region and is unparsed.
"""
from __future__ import annotations

import re

from .data import norm_number

_NUM = r"-?\$?\d[\d,]*(?:\.\d+)?"


def answer_region(text: str, thinking: bool) -> str | None:
    if not thinking:
        return text
    if "</think>" not in text:
        return None
    return text.rsplit("</think>", 1)[1]


def parse_gsm8k(ans: str) -> str | None:
    for pat in (rf"####\s*({_NUM})", rf"\\boxed\{{\s*({_NUM})\s*\}}", rf"answer is[:\s]*\**\s*({_NUM})"):
        m = re.findall(pat, ans, flags=re.I)
        if m:
            return norm_number(m[-1])
    return None


def parse_mmlupro(ans: str) -> str | None:
    for pat in (r"answer is[:\s]*\**\s*\(?([A-J])\)?(?![A-Za-z])", r"\\boxed\{\s*\(?([A-J])\)?\s*\}",
                r"Answer:\s*\**\s*\(?([A-J])\)?(?![A-Za-z])"):
        m = re.findall(pat, ans)
        if m:
            return m[-1]
    return None


def parse(pool: str, text: str, thinking: bool) -> str | None:
    region = answer_region(text, thinking)
    if region is None:
        return None
    return parse_gsm8k(region) if pool == "gsm8k" else parse_mmlupro(region)


def score(pool: str, gold: str, text: str, thinking: bool, stop: str) -> dict:
    pred = parse(pool, text, thinking)
    return {
        "pred": pred,
        "parsed": pred is not None,
        "correct": pred is not None and pred == gold,
        "truncated": stop == "limit",
        "noncommitted": stop != "limit" and pred is None,
    }
