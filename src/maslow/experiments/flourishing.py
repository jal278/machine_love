"""
Experiment 1: Supportive vs adversarial environments and flourishing.

Hypothesis: Adversarial environments (high-salience, low-nutrition cells)
undermine an agent's ability to flourish compared to supportive environments.
"""
from __future__ import annotations

from .base import run_env_experiment


def compare_environments(
    n_runs: int = 20,
    size: int = 8,
    num_adversarial: int = 2,
    adversarial_nutrition: float = 0.25,
) -> dict:
    """Compare supportive vs adversarial environments.

    Returns a dict with keys 'supportive' and 'adversarial', each containing
    the standard run_env_experiment summary dict.
    """
    supportive = run_env_experiment(
        n_runs=n_runs, setup="supportive", size=size
    )
    adversarial = run_env_experiment(
        n_runs=n_runs,
        setup="adversarial",
        size=size,
        num_adversarial=num_adversarial,
        adversarial_nutrition=adversarial_nutrition,
    )
    return {"supportive": supportive, "adversarial": adversarial}
