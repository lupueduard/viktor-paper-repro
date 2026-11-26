"""Rendering of requests for Qwen3 (ChatML) as raw completion prompts.

Thinking on: the assistant turn is left open and the model writes '<think> ... </think>' itself.
Thinking off: the empty think block that Qwen3's chat template inserts for enable_thinking=False
is pre-filled, so everything the model writes is the answer.
"""
from __future__ import annotations

from dataclasses import dataclass

from .data import LETTERS, Item

FORMAT = {
    "gsm8k": "Solve the problem. Write the final answer as a number on the last line in the form '#### <number>'.",
    "mmlupro": "Answer the multiple-choice question. End your reply with 'The answer is (X)', where X is the letter of the correct option.",
}
EMPTY_THINK = "<think>\n\n</think>\n\n"


@dataclass(frozen=True)
class Harness:
    """Everything around the frozen model that the study varies."""
    name: str
    thinking: bool
    max_tokens: int
    system: str = ""        # searched component 1
    instruction: str = ""   # searched component 2, appended to the user turn after a blank line


def user_message(item: Item) -> str:
    if item.pool == "gsm8k":
        body = item.question
    else:
        opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(item.options))
        body = f"Question: {item.question}\nOptions:\n{opts}"
    return f"{body}\n\n{FORMAT[item.pool]}"


def render(item: Item, h: Harness, leaked_reference: str | None = None) -> str:
    """Raw prompt. `leaked_reference` reproduces defect A: the row's reference answer sent as an
    earlier assistant turn before the generation prompt."""
    parts = []
    if h.system:
        parts.append(f"<|im_start|>system\n{h.system}<|im_end|>\n")
    user = user_message(item)
    if h.instruction:
        user += "\n\n" + h.instruction
    parts.append(f"<|im_start|>user\n{user}<|im_end|>\n")
    if leaked_reference is not None:
        parts.append(f"<|im_start|>assistant\n{leaked_reference}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    if not h.thinking:
        parts.append(EMPTY_THINK)
    return "".join(parts)


def reference_text(item: Item) -> str:
    """What a buggy caller would send as the assistant turn: the dataset's reference answer."""
    if item.pool == "gsm8k":
        return item.raw_answer
    return f"The answer is ({item.gold})"


# The grid (paper Table 1, scaled to what one 16-CPU box runs in a few hours).
THINK_64 = Harness("THINK-64", True, 64)
THINK_512 = Harness("THINK-512", True, 512)
OFF_1500 = Harness("OFF-1500", False, 1500)          # paper: AUTHORS-1500 (thinking off)
BUDGET_ANSWER = 1500                                   # BUDGET-512-1500: answer allowance after forced end
GRID = [THINK_64, THINK_512, OFF_1500]
