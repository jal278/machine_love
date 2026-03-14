"""Tests for experiment helpers (mocked runs)."""
import random
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from maslow.experiments.base import extract_data, run_env_experiment
from maslow.gridworld.base import MaslowAgent, MaslowGridworld


def _make_agent_with_log(n_steps: int = 10) -> MaslowAgent:
    """Build a minimal agent with a fake log for testing."""
    random.seed(0)
    world = MaslowGridworld(6, 6, (0, 0), setup="supportive")
    agent = MaslowAgent(world)
    for _ in range(n_steps):
        action = agent.update()
        world.step(action)
    return agent


def test_extract_data_keys():
    agent = _make_agent_with_log(20)
    data = extract_data(agent)
    assert "x" in data
    assert "engagement" in data
    assert "need" in data
    assert len(data["x"]) == 20
    assert len(data["engagement"]) == 20
    assert len(data["need"]) == 20


def test_extract_data_engagement_binary():
    agent = _make_agent_with_log(50)
    data = extract_data(agent)
    assert all(v in (0, 1) for v in data["engagement"])


def test_run_env_experiment_structure():
    random.seed(1)
    result = run_env_experiment(n_runs=2, setup="supportive", size=6, steps=50)
    assert "avg_need" in result
    assert "avg_engagement" in result
    assert "raw" in result
    assert len(result["raw"]) == 2


def test_run_env_experiment_adversarial():
    random.seed(2)
    result = run_env_experiment(
        n_runs=2, setup="adversarial", size=6, steps=50,
        num_adversarial=1, adversarial_nutrition=0.25
    )
    assert result["avg_need"] >= 0
    assert 0.0 <= result["avg_engagement"] <= 1.0
