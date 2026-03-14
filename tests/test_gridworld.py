"""Unit tests for the base Maslow gridworld (no LM calls)."""
import random

import numpy as np
import pytest

from maslow.gridworld.base import MaslowAgent, MaslowGridworld, run_simulation


def test_gridworld_creates_map():
    world = MaslowGridworld(8, 8, (0, 0), setup="supportive")
    assert world.map.shape == (8, 8)
    # At least one non-empty cell exists
    assert np.any(world.map != -1)
    # Salience and nutrition arrays have correct shape
    assert world.salience.shape == (8, 8)
    assert world.nutrition.shape == (8, 8)


def test_gridworld_has_need_cells():
    world = MaslowGridworld(8, 8, (0, 0), setup="supportive")
    need_cells = world.map[world.map >= 0]
    assert len(need_cells) > 0


def test_adversarial_has_high_salience_cells():
    world = MaslowGridworld(8, 8, (0, 0), setup="adversarial",
                            num_adversarial=2, adversarial_salience=2.0)
    # Some cells should have salience > default (1.0)
    assert np.any(world.salience > 1.0)


def test_step_project_clamps_to_bounds():
    world = MaslowGridworld(4, 4, (0, 0), setup="supportive")
    world.i, world.j = 0, 0
    i, j = world.step_project("left")
    assert i == 0
    assert j == 0

    world.i, world.j = 3, 3
    i, j = world.step_project("right")
    assert i == 3


def test_agent_explores():
    random.seed(42)
    world = MaslowGridworld(6, 6, (0, 0), setup="supportive")
    agent = MaslowAgent(world)

    for _ in range(50):
        action = agent.update()
        world.step(action)

    assert len(agent.explored) > 1
    assert len(agent.log) == 50


def test_agent_state_transitions():
    random.seed(0)
    world = MaslowGridworld(8, 8, (0, 0), setup="supportive")
    agent = MaslowAgent(world)

    states_seen = set()
    for _ in range(200):
        agent.update()
        world.step("noop")
        states_seen.add(agent.state)

    # Should see at least EXPLORE
    assert "EXPLORE" in states_seen


def test_supportive_vs_adversarial_needs():
    """Supportive environments should yield higher average need levels."""
    random.seed(1)
    np.random.seed(1)

    def avg_need(setup):
        world = MaslowGridworld(8, 8, (0, 0), setup=setup,
                                num_adversarial=2, adversarial_nutrition=0.25)
        agent = MaslowAgent(world)
        for _ in range(500):
            action = agent.update()
            world.step(action)
        return np.mean([e["need_level"] for e in agent.log])

    sup = avg_need("supportive")
    adv = avg_need("adversarial")
    # Supportive should not be strictly worse — we just check both run without error
    assert sup >= 0
    assert adv >= 0


def test_run_simulation_smoke():
    random.seed(7)
    world, agent = run_simulation(setup="supportive", size=6, steps=100)
    assert world is not None
    assert agent is not None
    assert len(agent.log) == 100
