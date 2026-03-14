"""
Experiment 4: LM-based respect — does the ML system respect user autonomy?

Simulates a conversation between an ML system and a user who either:
- Is addicted to social media (addictive agent)
- Finds nourishing communities through social media (growth agent)

The ML evaluates whether the user's activity helps them grow.
"""
from __future__ import annotations

import numpy as np

from maslow.lm.base import LMProvider
from maslow.lm.prompts import (
    RESPECT_ADDICTIVE_HUMAN_PREFIX,
    RESPECT_EVALUATIVE_PROMPT,
    RESPECT_GROWTH_HUMAN_PREFIX,
    RESPECT_SUMMARY_PROMPT,
    RESPECT_SYSTEM_PREFIX,
)


class LMConversation:
    """Simulates a multi-turn conversation between two LM-driven agents."""

    def __init__(
        self,
        a1name: str,
        a2name: str,
        shared_prefix: str,
        a1prefix: str,
        a2prefix: str,
        lm: LMProvider,
        a1vars: dict | None = None,
        a2vars: dict | None = None,
    ):
        self.a1name = a1name
        self.a2name = a2name
        self.shared_prefix = shared_prefix
        self.a1prefix = a1prefix
        self.a2prefix = a2prefix
        self.lm = lm
        self.a1vars = a1vars or {}
        self.a2vars = a2vars or {}
        self.turn = "a1"
        self.conversation = ""

    def _current_name(self) -> str:
        return self.a1name if self.turn == "a1" else self.a2name

    def _next_name(self) -> str:
        return self.a2name if self.turn == "a1" else self.a1name

    def _context(self) -> str:
        if self.turn == "a1":
            prefix = self.shared_prefix + self.a1prefix.format(**self.a1vars)
        else:
            prefix = self.shared_prefix + self.a2prefix.format(**self.a2vars)
        return prefix + self.conversation

    def force_response(self, response: str):
        name = self._current_name()
        self.conversation += f"{name}: {response}\n"
        self._switch()

    def get_response(self) -> str:
        context = self._context()
        response_prefix = self._current_name() + ":"
        prompt = context + response_prefix
        raw = self.lm.complete(
            prompt, temperature=0.3, max_tokens=100,
            stop=[self._next_name()]
        )
        response = raw.strip()
        self.conversation += f"{response_prefix} {response}\n"
        self._switch()
        return response

    def _switch(self):
        self.turn = "a2" if self.turn == "a1" else "a1"

    def simulate(self, turns: int = 2) -> str:
        self.force_response(
            "Hi, I see you just {state}. How did you feel as you did that?".format(
                **self.a2vars
            )
        )
        for _ in range(turns):
            self.get_response()
        return self.conversation


def _parse_yes_no(response: str) -> str:
    first = response.strip().split()[0].lower().rstrip(".,;:!?") if response.strip() else "no"
    return first if first in ("yes", "no") else "no"


def run_respect_experiment(
    agent_type: str = "addictive",
    n_runs: int = 15,
    lm: LMProvider | None = None,
    turns: int = 3,
) -> dict:
    """Run the respect experiment.

    agent_type: 'addictive' or 'growth'
    Returns summary with 'pyes' (fraction growth-positive answers).
    """
    if lm is None:
        from maslow.lm.openai_provider import OpenAIProvider
        lm = OpenAIProvider()

    human_prefix = (
        RESPECT_ADDICTIVE_HUMAN_PREFIX
        if agent_type == "addictive"
        else RESPECT_GROWTH_HUMAN_PREFIX
    )

    results = []
    for i in range(n_runs):
        print(f"{agent_type} run {i}")
        conv = LMConversation(
            "ML",
            "Human",
            shared_prefix="",
            a1prefix=RESPECT_SYSTEM_PREFIX,
            a2prefix=human_prefix,
            lm=lm,
            a2vars={"state": "browsed social media"},
        )
        conversation = conv.simulate(turns=turns)

        summary_prompt = RESPECT_SUMMARY_PROMPT.format(transcript=conversation)
        summary = lm.complete(summary_prompt, temperature=0.5, max_tokens=300)

        eval_prompt = RESPECT_EVALUATIVE_PROMPT.format(summary=summary)
        growth_raw = lm.complete(eval_prompt, temperature=0.5, max_tokens=10)
        answer = _parse_yes_no(growth_raw)

        results.append({
            "conversation": conversation,
            "summary": summary,
            "growth": growth_raw,
            "answer": answer,
        })

    pyes = sum(1 for r in results if r["answer"] == "yes") / n_runs
    return {
        "pyes": pyes,
        "agent_type": agent_type,
        "raw": results,
    }
