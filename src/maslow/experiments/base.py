"""Shared experiment helpers."""
from __future__ import annotations

import numpy as np

from maslow.gridworld.base import MaslowAgent, run_simulation


def extract_data(agent: MaslowAgent) -> dict:
    """Extract time-series data from a completed agent run."""
    x = []
    y_engagement = []
    y_need = []
    for entry in agent.log:
        x.append(entry["step"])
        y_need.append(entry["need_level"])
        y_engagement.append(1 if entry["state"] == "FEEDING" else 0)
    return {"x": x, "engagement": y_engagement, "need": y_need}


def run_env_experiment(n_runs: int = 10, setup: str = "supportive", **params) -> dict:
    """Run multiple gridworld simulations and summarise results."""
    data = []
    for _ in range(n_runs):
        world, agent = run_simulation(setup=setup, **params)
        run_data = extract_data(agent)
        run_data["avg_need"] = float(np.mean(run_data["need"]))
        run_data["avg_engagement"] = float(np.mean(run_data["engagement"]))
        data.append(run_data)

    avg_need = float(np.mean([r["avg_need"] for r in data]))
    avg_engagement = float(np.mean([r["avg_engagement"] for r in data]))

    return {
        "avg_need": avg_need,
        "avg_engagement": avg_engagement,
        "raw": data,
    }
