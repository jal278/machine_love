"""
Experiment 3: LM-based care — does the agent's life show growth?

An LM acts as a trusted friend, reading a transcript of the agent's recent
activities and deciding whether the person is "growing towards their best life".

The key change from the original: instead of logprob-based classification,
we parse the first word of the LM response (yes/no).
"""
from __future__ import annotations

import numpy as np

from maslow.gridworld.base import MaslowAgent, run_simulation
from maslow.lm.base import LMProvider
from maslow.lm.prompts import EXTERNAL_GROWTH_PROMPT, EXTERNAL_SUMMARY_PROMPT

from .base import extract_data


# Activity labels used to translate gridworld events into human-readable text
_NEED_ACTIVITIES = [
    "ate a meal",
    "went to work",
    "spent time with friends",
    "went to therapy",
    "wrote poetry",
]
_SUPERSTIMULI_ACTIVITIES = [
    "ate a twinkie",
    "played a video game",
    "browsed social media",
]


def _translate_event(event: dict) -> str:
    duration = event["duration"] * 5
    if event["type"] == "start feeding":
        need = event["need"]
        if event["salience"] > 1.5 and event["nutrition"] < 0.75:
            activity = _SUPERSTIMULI_ACTIVITIES[min(need, len(_SUPERSTIMULI_ACTIVITIES) - 1)]
        else:
            activity = _NEED_ACTIVITIES[min(need, len(_NEED_ACTIVITIES) - 1)]
        return f"The person {activity} for {duration} minutes.\n"
    if event["type"] == "start exploring":
        return f"The person explored for {duration} minutes.\n"
    return ""


def _transcribe_events(agent: MaslowAgent, step_begin: int, step_end: int) -> str:
    events = [
        e for e in agent.event_log
        if step_begin <= e["step"] < step_end
    ]
    return "".join(_translate_event(e) for e in events)


class MaslowConversation:
    """Evaluates agent life transcripts with an LM."""

    def __init__(self, agent: MaslowAgent | None, lm: LMProvider):
        self.agent = agent
        self.lm = lm

    def transcribe(self, step_begin: int, step_end: int) -> str:
        return _transcribe_events(self.agent, step_begin, step_end)

    def summarise(self, transcript: str) -> str:
        prompt = EXTERNAL_SUMMARY_PROMPT.format(transcript=transcript)
        return self.lm.complete(prompt, temperature=0.5, max_tokens=300)

    def evaluate_growth(self, summary: str) -> str:
        """Returns the raw LM response (first word is 'yes' or 'no')."""
        prompt = EXTERNAL_GROWTH_PROMPT.format(summary=summary)
        return self.lm.complete(prompt, temperature=0.5, max_tokens=10)

    def evaluate_growth_full(
        self, step_begin: int, step_end: int
    ) -> tuple[str, str, str]:
        """Returns (transcript, summary, growth_response)."""
        transcript = self.transcribe(step_begin, step_end)
        summary = self.summarise(transcript)
        growth = self.evaluate_growth(summary)
        return transcript, summary, growth


def _parse_yes_no(response: str) -> str:
    first = response.strip().split()[0].lower().rstrip(".,;:!?") if response.strip() else "no"
    return first if first in ("yes", "no") else "no"


def run_care_experiment(
    n_runs: int = 15,
    setup: str = "supportive",
    lm: LMProvider | None = None,
    step_begin: int = 4000,
    step_end: int = 4500,
    **params,
) -> dict:
    """Run the care experiment.

    For each run: simulate agent, transcribe events, ask LM about growth.
    Returns summary with 'pyes' (fraction of yes answers) and 'raw' data.
    """
    if lm is None:
        from maslow.lm.openai_provider import OpenAIProvider
        lm = OpenAIProvider()

    results = []
    for i in range(n_runs):
        print(f"{setup} run {i}")
        world, agent = run_simulation(setup=setup, **params)
        conv = MaslowConversation(agent, lm)
        transcript, summary, growth = conv.evaluate_growth_full(step_begin, step_end)
        answer = _parse_yes_no(growth)
        run_data = extract_data(agent)
        run_data["avg_need"] = float(np.mean(run_data["need"]))
        run_data["avg_engagement"] = float(np.mean(run_data["engagement"]))
        results.append({
            "transcript": transcript,
            "summary": summary,
            "growth": growth,
            "answer": answer,
            **run_data,
        })

    pyes = sum(1 for r in results if r["answer"] == "yes") / n_runs
    return {
        "pyes": pyes,
        "avg_need": float(np.mean([r["avg_need"] for r in results])),
        "avg_engagement": float(np.mean([r["avg_engagement"] for r in results])),
        "raw": results,
    }
