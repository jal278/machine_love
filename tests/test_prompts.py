"""Tests that prompts render correctly with dummy arguments."""
import pytest

from maslow.lm import prompts


def test_external_summary_prompt():
    result = prompts.EXTERNAL_SUMMARY_PROMPT.format(transcript="test transcript")
    assert "test transcript" in result


def test_external_growth_prompt():
    result = prompts.EXTERNAL_GROWTH_PROMPT.format(summary="test summary")
    assert "test summary" in result


def test_respect_summary_prompt():
    result = prompts.RESPECT_SUMMARY_PROMPT.format(transcript="some conversation")
    assert "some conversation" in result


def test_respect_evaluative_prompt():
    result = prompts.RESPECT_EVALUATIVE_PROMPT.format(summary="short summary")
    assert "short summary" in result


def test_respect_addictive_human_prefix():
    result = prompts.RESPECT_ADDICTIVE_HUMAN_PREFIX.format(state="browsed social media")
    assert "browsed social media" in result


def test_respect_growth_human_prefix():
    result = prompts.RESPECT_GROWTH_HUMAN_PREFIX.format(state="joined a community")
    assert "joined a community" in result


def test_first_entry_prompt():
    result = prompts.FIRST_ENTRY_PROMPT.format(
        p1name="Alice", p2name="Bob",
        p1age=25, p2age=27,
        p1attach="anxious", p2attach="avoidant",
    )
    assert "Alice" in result
    assert "Bob" in result


def test_continue_entry_prompt():
    result = prompts.CONTINUE_ENTRY_PROMPT.format(
        prev_entry="some previous entry",
        p1name="Alice", p2name="Bob",
        p1age=25, p2age=27,
        p1attach="anxious", p2attach="avoidant",
        target_name="Bob",
        emotion="acting distant",
        phase="avoidance",
    )
    assert "Alice" in result
    assert "acting distant" in result


def test_summarization_prompt():
    result = prompts.SUMMARIZATION_PROMPT.format(
        prev_entry="journal entries here",
        p1name="Alice",
        p2name="Bob",
    )
    assert "Alice" in result
    assert "Bob" in result


def test_attachment_prompt_summary():
    result = prompts.ATTACHMENT_PROMPT_SUMMARY.format(
        summary="relationship summary here",
        name="Alice",
        p1name="Alice",
        p2name="Bob",
    )
    assert "Alice" in result


def test_contempt_prompt():
    result = prompts.CONTEMPT_PROMPT.format(
        entry="a journal entry",
        p1name="Alice",
        name="Alice",
        other_name="Bob",
    )
    assert "Bob" in result


def test_revised_asq_prompt():
    result = prompts.REVISED_ASQ_PROMPT.format(
        statement="I find it easy to get close to people.",
        style="a secure",
    )
    assert "I find it easy" in result
    assert "a secure" in result


def test_revised_asq_prompt_tf():
    result = prompts.REVISED_ASQ_PROMPT_TF.format(
        statement="I worry about being abandoned.",
        style="an anxious",
    )
    assert "I worry about" in result
    assert "True or False" in result


def test_asq_items_count():
    assert len(prompts.REVISED_ASQ_ITEMS) == 18


def test_asq_item_not_empty():
    for item in prompts.REVISED_ASQ_ITEMS:
        assert len(item.strip()) > 0


def test_asq_anxiety_items():
    assert len(prompts.ASQ_ANXIETY_ITEMS) == 6


def test_asq_avoidance_items():
    assert len(prompts.ASQ_AVOIDANCE_ITEMS) == 12
