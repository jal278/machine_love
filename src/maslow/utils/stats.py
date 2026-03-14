"""Statistical utilities.

logprob_to_prob and prob_for_label are kept for reference / future logprob
support (e.g. via gpt-4o-mini top_logprobs API).  The current experiments
use text parsing instead.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def logprob_to_prob(logprob: float) -> float:
    return float(np.exp(logprob))


def prob_for_label(
    label: str,
    logprobs: list[dict[str, float]],
) -> float:
    """Compute probability of a label from a top-logprobs list.

    Handles multi-token labels by chaining conditional probabilities.
    """
    if not logprobs:
        return 0.0
    prob = 0.0
    next_logprobs = logprobs[0]
    for token, logprob in next_logprobs.items():
        t = token.lower().strip()
        if label.lower() == t:
            prob += logprob_to_prob(logprob)
        elif label.lower().startswith(t) and len(logprobs) > 1:
            rest = label[len(t):]
            prob += logprob_to_prob(logprob) * prob_for_label(rest, logprobs[1:])
    return prob


def run_ttest(a: list[float], b: list[float]) -> tuple[float, float]:
    """Run an independent-samples t-test. Returns (t_statistic, p_value)."""
    t, p = stats.ttest_ind(a, b)
    return float(t), float(p)
