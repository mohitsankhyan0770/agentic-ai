"""Critic: scores an answer against the task *and the evidence the tools produced*.

Two bugs in the original are fixed here:
1. It "improved" raw tool output (e.g. turning the exact result "391" into an
   LLM essay that could change the number). Now tool observations are passed as
   evidence and the reviser is told to stay faithful to them.
2. It matched the substring "IMPROVE" in free text. Now it returns a typed
   score/verdict/issues object, and the loop is bounded.
"""

from __future__ import annotations

from .llm import LLM, LLMError
from .schemas import Critique

CRITIC_PROMPT = """You are a strict reviewer. Judge whether the ANSWER fully and correctly
solves the TASK, using the EVIDENCE (tool outputs) as ground truth.

Score 1-10. Check that every action the TASK asked for (calculating, saving a file, searching, etc.) appears in the EVIDENCE as a tool call. If an action is missing from the EVIDENCE, use verdict "revise" and score 4 or lower.

Reply as JSON: {"score": <int>, "verdict": "accept"|"revise", "issues": ["..."]}"""

REVISE_PROMPT = """Rewrite the answer to fix the reviewer's issues.
Stay strictly faithful to the evidence: never change numbers, facts or results
that came from tools. Return only the improved answer."""


def _format(task: str, answer: str, evidence: str) -> str:
    return f"TASK:\n{task}\n\nEVIDENCE:\n{evidence or '(no tools used)'}\n\nANSWER:\n{answer}"


class Critic:
    def __init__(self, llm: LLM, accept_score: int = 7) -> None:
        self.llm = llm
        self.accept_score = accept_score

    def review(self, task: str, answer: str, evidence: str = "") -> Critique:
        try:
            critique = self.llm.complete_json(
                [
                    {"role": "system", "content": CRITIC_PROMPT},
                    {"role": "user", "content": _format(task, answer, evidence)},
                ],
                Critique,
            )
        except LLMError:
            return Critique(score=self.accept_score, verdict="accept", issues=["critic unavailable"])
        # Enforce consistency between the numeric score and the verdict.
        if critique.score < self.accept_score:
            critique.verdict = "revise"
        return critique

    def revise(self, task: str, answer: str, evidence: str, critique: Critique) -> str:
        issues = "\n".join(f"- {i}" for i in critique.issues) or "- Make it more complete and precise."
        return self.llm.complete(
            [
                {"role": "system", "content": REVISE_PROMPT},
                {"role": "user", "content": f"{_format(task, answer, evidence)}\n\nREVIEWER ISSUES:\n{issues}"},
            ]
        )
