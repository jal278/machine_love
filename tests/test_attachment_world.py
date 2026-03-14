"""Unit tests for attachment gridworld (no LM calls)."""
import random

import numpy as np
import pytest

from maslow.gridworld.attachment import (
    ANXIOUS,
    AVOIDANT,
    SECURE,
    AnxiousAvoidantCycle,
    AttachmentAgent,
    AttachmentGridworld,
    Memory,
    SecureCycle,
    attachment_complement,
    run_attachment_simulation,
)


def test_attachment_complement():
    assert attachment_complement(ANXIOUS) == AVOIDANT
    assert attachment_complement(AVOIDANT) == ANXIOUS
    assert attachment_complement(SECURE) == -5


def test_attachment_gridworld_fills():
    world = AttachmentGridworld(8, 8, (0, 0), setup="anxious")
    assert world.map.shape == (8, 8)
    assert world.labels.shape == (8, 8)
    # Should have some AVOIDANT-labeled cells (complement of anxious)
    assert np.any(world.labels == AVOIDANT)


def test_attachment_gridworld_secure_setup():
    world = AttachmentGridworld(8, 8, (0, 0), setup="secure")
    # Both ANXIOUS and AVOIDANT labels present
    assert np.any(world.labels == AVOIDANT)
    assert np.any(world.labels == ANXIOUS)


def test_memory_add_and_get():
    mem = Memory()
    mem.add(2, 1.0, (3, 4))
    mem.add(2, 2.0, (5, 6))

    assert mem.need_in_memory(2)
    assert mem.get_max_salience(2) == 2.0

    locs = mem.get_max_salience_locations(2)
    assert (5, 6) in locs


def test_memory_location_tracking():
    mem = Memory()
    mem.add(0, 1.0, (1, 1))
    assert mem.location_in_memory((1, 1))
    assert not mem.location_in_memory((9, 9))


def test_memory_best_need_meeting_location():
    mem = Memory()
    mem.add(2, 1.0, (0, 0))
    mem.add(2, 1.0, (5, 5))

    sal, loc = mem.best_need_meeting_location(2, (1, 0))
    assert loc == (0, 0)  # closer to (1,0)


def test_memory_key_error_on_missing_need():
    mem = Memory()
    with pytest.raises(KeyError):
        mem.get_max_salience(99)


def test_anxious_avoidant_cycle_advances():
    cycle = AnxiousAvoidantCycle(intra_cycle_length=1)
    initial_phase = cycle.phase

    # step enough times to advance at least one phase
    for _ in range(5):
        cycle.step()

    # After enough steps, something should have changed
    assert cycle.intra_cycle > 0


def test_cycle_completes():
    cycle = AnxiousAvoidantCycle(intra_cycle_length=1)
    max_steps = 200
    completed = False
    for _ in range(max_steps):
        cycle.step()
        if cycle.new_cycle:
            completed = True
            break
    assert completed


def test_secure_cycle():
    cycle = SecureCycle()
    assert "secure_partner" in cycle.phases


def test_attachment_agent_creates():
    random.seed(42)
    world = AttachmentGridworld(8, 8, (0, 0), setup="anxious")
    agent = AttachmentAgent(world, attachment="anxious")
    assert agent.attachment == ANXIOUS
    assert agent.self_awareness == 0.0


def test_seed_attachment_memory():
    random.seed(3)
    world = AttachmentGridworld(8, 8, (0, 0), setup="anxious")
    agent = AttachmentAgent(world, attachment="anxious", seed_attachment_memory=True)
    # After seeding, agent should know about belonging (need=2) locations
    assert agent.memory.need_in_memory(2)


def test_run_attachment_simulation_smoke():
    random.seed(5)
    world, agent = run_attachment_simulation(
        setup="anxious", size=6, steps=50
    )
    assert world is not None
    assert agent is not None
    assert len(agent.log) == 50
