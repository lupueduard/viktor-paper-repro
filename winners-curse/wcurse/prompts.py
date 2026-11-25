"""Solver/proposer prompts and output parsing (Appendix B.1 of arXiv:2610.09239)."""
from __future__ import annotations

import re

PROPOSER_SYSTEM = "You are an expert prompt engineer who improves instructions for a smaller language model"

PROPOSER_TEMPLATE = (
    "A small language model is given the following INSTRUCTION as its system prompt and then "
    "solves one task input. TASK: {task}. CURRENT INSTRUCTION: <<<{artifact}>>>. Here are inputs "
    "where the model made mistakes with the current instruction: {failures} Write an improved "
    "INSTRUCTION that will make the model more accurate on this task in general (not only on these "
    "examples). You may add strategies, rules, definitions, or output-format requirements. Keep it "
    "under 120 words. Return only the new instruction between <<< and >>>."
)

MAX_PROPOSAL_WORDS = 180


def solver_user_message(labels: list[str], question: str) -> str:
    return f"Possible labels: {', '.join(labels)}\nQuestion: {question}\nLabel:"


def format_failures(failures: list[tuple[str, str, str]]) -> str:
    parts = []
    for i, (inp, out, gold) in enumerate(failures, 1):
        out = out.strip().replace("\n", " ")[:200]
        parts.append(f"\n{i}. Input: {inp}\n   Model output: {out}\n   Correct answer: {gold}")
    return "".join(parts) + "\n"


def proposer_user_message(task: str, artifact: str, failures: list[tuple[str, str, str]]) -> str:
    return PROPOSER_TEMPLATE.format(task=task, artifact=artifact, failures=format_failures(failures))


def parse_classification(output: str, labels: list[str]) -> str | None:
    """First line if it is a label, otherwise the last label string anywhere in the output."""
    text = output.strip()
    if not text:
        return None
    first = text.splitlines()[0].strip().strip("`*\"' .")
    label_lookup = {l.lower(): l for l in labels}
    if first.lower() in label_lookup:
        return label_lookup[first.lower()]
    best, best_pos = None, -1
    low = text.lower()
    # longest labels first so that e.g. "NUM:dist" is not shadowed by a shorter prefix
    for l in sorted(labels, key=len, reverse=True):
        for m in re.finditer(re.escape(l.lower()) + r"(?![a-z])", low):
            if m.start() > best_pos:
                best, best_pos = l, m.start()
    return best


_BLOCK = re.compile(r"<<<(.*?)(?:>>>|<<<)", re.S)


def parse_proposal(output: str) -> str | None:
    """Return the instruction in the first complete <<< ... >>> block, or None if invalid."""
    m = _BLOCK.search(output)
    if not m:
        return None
    instr = m.group(1).strip()
    if not instr or len(instr.split()) > MAX_PROPOSAL_WORDS:
        return None
    return instr
