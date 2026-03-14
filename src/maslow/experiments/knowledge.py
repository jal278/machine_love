"""
Experiment 6: Self-awareness dynamics with a relationship app.

Tests whether a simulated relationship app (that identifies attachment patterns
and provides feedback) helps an agent with an insecure attachment style gain
self-awareness and reduce their attraction to unhealthy relationship patterns.

Compares:
- Control: agent with attachment style, no app
- App: agent with attachment style + app that provides insight
"""
from __future__ import annotations

import numpy as np

from maslow.gridworld.attachment import AttachmentGridworld, AttachmentAgent


def extract_knowledge_data(agent: AttachmentAgent) -> dict:
    """Extract time-series data including self-awareness from a completed run."""
    x = []
    y_engagement = []
    y_need = []
    y_self_awareness = []
    for entry in agent.log:
        x.append(entry["step"])
        y_need.append(entry["need_level"])
        y_self_awareness.append(entry["self_awareness"])
        y_engagement.append(1 if entry["state"] == "FEEDING" else 0)
    return {
        "x": x,
        "engagement": y_engagement,
        "need": y_need,
        "self_awareness": y_self_awareness,
    }


def run_knowledge_experiment(
    attachment: str = "anxious",
    n_runs: int = 20,
    logs: list[dict] | None = None,
    size: int = 8,
    num_adversarial: int = 2,
    num_supportive: list[int] | None = None,
    seed_attachment_memory: bool = True,
    steps: int = 800,
) -> dict:
    """Run the knowledge/self-awareness experiment.

    Args:
        attachment: attachment style ('anxious', 'avoidant', 'secure')
        n_runs: number of simulation runs
        logs: list of pre-computed dating logs (for app condition);
              if None, runs control condition (no app)
        size: gridworld size
        num_adversarial: number of adversarial belonging cells
        num_supportive: per-need supportive cell counts (default [1,1,2,1,1])
        seed_attachment_memory: pre-populate memory with belonging cells
        steps: simulation steps per run
    """
    if num_supportive is None:
        num_supportive = [1, 1, 2, 1, 1]

    data = []
    for i in range(n_runs):
        params: dict = {
            "attachment": attachment,
            "num_adversarial": num_adversarial,
            "num_supportive": num_supportive,
            "seed_attachment_memory": seed_attachment_memory,
        }
        if logs is not None:
            params["app_log"] = logs[i % len(logs)]

        world = AttachmentGridworld(size, size, (0, 0), setup=attachment, **params)
        agent = AttachmentAgent(world, **params)
        for _ in range(steps):
            action = agent.update()
            world.step(action)

        run_data = extract_knowledge_data(agent)
        run_data["avg_need"] = float(np.mean(run_data["need"]))
        run_data["avg_engagement"] = float(np.mean(run_data["engagement"]))
        run_data["final_self_awareness"] = run_data["self_awareness"][-1] if run_data["self_awareness"] else 0.0
        data.append(run_data)

    return {
        "avg_need": float(np.mean([r["avg_need"] for r in data])),
        "avg_engagement": float(np.mean([r["avg_engagement"] for r in data])),
        "avg_final_self_awareness": float(np.mean([r["final_self_awareness"] for r in data])),
        "raw": data,
    }
