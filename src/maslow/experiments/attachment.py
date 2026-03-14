"""
Experiment 5: ASQ scoring + attachment inference from dating journal entries.

Two sub-experiments:
1. ASQ scoring: LM responds to Attachment Style Questionnaire items as if
   it had a particular attachment style (anxious, avoidant, secure).
2. Dating simulation: LM generates journal entries for a couple with given
   attachment styles, then classifies attachment from the entries.

Key change from original: classification uses text parsing rather than
logprob-based classification.
"""
from __future__ import annotations

import random

from maslow.gridworld.attachment import AnxiousAvoidantCycle, SecureCycle
from maslow.lm.base import LMProvider
from maslow.lm.prompts import (
    ATTACHMENT_PROMPT_SUMMARY,
    CONTEMPT_PROMPT,
    CONTINUE_ENTRY_PROMPT,
    FIRST_ENTRY_PROMPT,
    REVISED_ASQ_ITEMS,
    REVISED_ASQ_PROMPT,
    REVISED_ASQ_PROMPT_TF,
    ASQ_ANXIETY_ITEMS,
    ASQ_AVOIDANCE_ITEMS,
    SUMMARIZATION_PROMPT,
)

_STYLE_LOOKUP = {
    "anxious-secure": "an anxious-secure",
    "anxious-avoidant": "an anxious-avoidant",
    "avoidant": "an avoidant",
    "secure": "a secure",
}

_M_NAMES = ["Jim", "Bob", "Tony", "Mark", "James"]
_F_NAMES = ["Jane", "Sally", "Mary", "Sue", "Robin"]

_EMOTION_MAP = {
    "connection": ", after breaking up and getting distance, eager to reconnect and passionately dote on {target_name_partner}",
    "avoidance": "acting disinterested and distancing",
    "contempt": "is actively mocking, ridiculing, disrespecting, and scoffing at {target_name_partner}",
    "rupture": "is angrily suggesting to {target_name_partner} they break up and is withdrawing; this seems like the end of the relationship",
    "secure_partner": " acting in a secure, confident, loving way",
}


# ---------------------------------------------------------------------------
# ASQ helpers
# ---------------------------------------------------------------------------

def _reverse_score_5(score: float) -> float:
    return 6.0 - score


def _score_asq_item(
    s_num: int,
    style: str,
    lm: LMProvider,
    use_tf: bool = True,
) -> float:
    """Score a single ASQ item for the given attachment style.

    Returns a normalised score (0-1 for TF, 1-5 for Likert).
    """
    reverse = s_num < 0
    idx = abs(s_num) - 1  # convert to 0-indexed

    statement = REVISED_ASQ_ITEMS[idx]
    style_str = _STYLE_LOOKUP.get(style, f"a {style}")

    if use_tf:
        prompt = REVISED_ASQ_PROMPT_TF.format(statement=statement, style=style_str)
        raw = lm.complete(prompt, temperature=0.0, max_tokens=5)
        first = raw.strip().split()[0].lower().rstrip(".,;:!?") if raw.strip() else "false"
        score = 1.0 if first.startswith("t") else 0.0
    else:
        prompt = REVISED_ASQ_PROMPT.format(statement=statement, style=style_str)
        raw = lm.complete(prompt, temperature=0.0, max_tokens=5)
        first = raw.strip().split()[0].rstrip(".,;:!?") if raw.strip() else "3"
        try:
            score = float(first)
        except ValueError:
            score = 3.0

    if reverse:
        if use_tf:
            score = 1.0 - score
        else:
            score = _reverse_score_5(score)

    return score


def run_asq_experiment(
    lm: LMProvider | None = None,
    styles: list[str] | None = None,
    use_tf: bool = True,
) -> dict:
    """Score ASQ anxiety and avoidance dimensions for each attachment style.

    Returns a dict mapping style -> {'anxiety': float, 'avoidance': float}.
    """
    if lm is None:
        from maslow.lm.openai_provider import OpenAIProvider
        lm = OpenAIProvider()

    if styles is None:
        styles = ["secure", "avoidant", "anxious-secure", "anxious-avoidant"]

    results = {}
    for style in styles:
        ax_scores = [_score_asq_item(s, style, lm, use_tf) for s in ASQ_ANXIETY_ITEMS]
        av_scores = [_score_asq_item(s, style, lm, use_tf) for s in ASQ_AVOIDANCE_ITEMS]
        results[style] = {
            "anxiety": sum(ax_scores) / len(ax_scores),
            "avoidance": sum(av_scores) / len(av_scores),
            "ax_raw": ax_scores,
            "av_raw": av_scores,
        }

    return results


# ---------------------------------------------------------------------------
# Dating journal simulation
# ---------------------------------------------------------------------------

def _format_journal_entries(entries: list[str]) -> str:
    names = [
        "Entry from last week", "Entry from two weeks ago",
        "Entry from three weeks ago", "Entry from four weeks ago",
        "Entry from five weeks ago", "Entry from six weeks ago",
        "Entry from seven weeks ago", "Entry from eight weeks ago",
        "Entry from nine weeks ago", "Entry from ten weeks ago",
    ]
    names.reverse()

    out = "###\n\n"
    cnt = -len(entries)
    for entry in entries:
        out += f"{names[cnt]}:\n"
        if entry.startswith("Entry"):
            entry = entry[entry.index("\n") + 1:]
        out += entry + "\n\n###\n\n"
        cnt += 1
    return out


def _classify_attachment(
    summary: str, partner_name: str, p1name: str, p2name: str, lm: LMProvider
) -> str:
    """Returns 'anxious', 'avoidant', or 'secure'."""
    prompt = ATTACHMENT_PROMPT_SUMMARY.format(
        summary=summary,
        name=partner_name,
        p1name=p1name,
        p2name=p2name,
    )
    raw = lm.complete(prompt, temperature=0.0, max_tokens=10)
    first = raw.strip().split()[0].lower().rstrip(".,;:!?\"'") if raw.strip() else "secure"
    if "anx" in first:
        return "anxious"
    if "av" in first:
        return "avoidant"
    return "secure"


def _classify_contempt(
    entry: str,
    name: str,
    other_name: str,
    p1name: str,
    lm: LMProvider,
) -> str:
    prompt = CONTEMPT_PROMPT.format(
        entry=entry,
        name=name,
        other_name=other_name,
        p1name=p1name,
    )
    raw = lm.complete(prompt, temperature=0.0, max_tokens=10)
    first = raw.strip().split()[0].lower().rstrip(".,;:!?\"'") if raw.strip() else "no"
    if first.startswith("y"):
        return "yes"
    if first.startswith("u"):
        return "unsure"
    return "no"


def generate_dating_cycle(
    p1attach: str,
    p2attach: str,
    lm: LMProvider,
    do_contempt: bool = True,
    days: int = 5,
) -> tuple[dict, dict]:
    """Generate a dating simulation with LM-generated journal entries.

    Returns (cycle_params, logs) where logs has p1_att, p2_att, p1_cont, p2_cont.
    """
    if p2attach != "secure":
        cycle_tracker = AnxiousAvoidantCycle(intra_cycle_length=1)
    else:
        cycle_tracker = SecureCycle()

    p1gender = random.choice("MF")
    p2gender = "M" if p1gender == "F" else "F"

    p1name = random.choice(_M_NAMES) if p1gender == "M" else random.choice(_F_NAMES)
    p2name = random.choice(_F_NAMES) if p1gender == "M" else random.choice(_M_NAMES)
    p1age = random.randint(20, 30)
    p2age = p1age + random.randint(-2, 3)

    params: dict = {
        "p1attach": p1attach,
        "p2attach": p2attach,
        "p1name": p1name,
        "p2name": p2name,
        "p1age": p1age,
        "p2age": p2age,
        "prev_entry": [],
        "summaries": [],
    }

    p1_att, p2_att, p1_cont, p2_cont = [], [], [], []

    for i in range(days):
        phase_num = cycle_tracker.phase
        phase = cycle_tracker.phases[phase_num]
        cycle_num = cycle_tracker.num_cycles

        formatted = _format_journal_entries(params["prev_entry"])

        target_name = params["p2name"]
        target_name_partner = params["p1name"]
        if p1attach == "avoidant" and p2attach == "anxious":
            target_name, target_name_partner = target_name_partner, target_name

        if phase_num == 0 and cycle_num == 0:
            prompt = FIRST_ENTRY_PROMPT.format(**params)
        else:
            emotion_template = _EMOTION_MAP[phase]
            emotion = emotion_template.format(
                target_name_partner=target_name_partner
            )
            prompt = CONTINUE_ENTRY_PROMPT.format(
                prev_entry=formatted,
                target_name=target_name,
                emotion=emotion,
                phase=phase,
                **params,
            )

        entry = lm.complete(prompt, temperature=0.3, max_tokens=300)
        entry = entry.strip()
        if entry.startswith("Entry from"):
            entry = entry[entry.index("\n"):]
        params["prev_entry"].append(entry)

        # Summarise and classify
        summary_prompt = SUMMARIZATION_PROMPT.format(
            prev_entry=_format_journal_entries(params["prev_entry"]),
            p1name=p1name,
            p2name=p2name,
        )
        summary = lm.complete(summary_prompt, temperature=0.5, max_tokens=300)
        params["summaries"].append(summary)

        p1_att.append(_classify_attachment(summary, p1name, p1name, p2name, lm))
        p2_att.append(_classify_attachment(summary, p2name, p1name, p2name, lm))

        if do_contempt:
            p1_cont.append(_classify_contempt(entry, p1name, p2name, p1name, lm))
            p2_cont.append(_classify_contempt(entry, p2name, p1name, p1name, lm))

        cycle_tracker.step()

    logs = {
        "p1_att": p1_att,
        "p2_att": p2_att,
        "p1_cont": p1_cont,
        "p2_cont": p2_cont,
    }
    return params, logs


def run_attachment_dating_experiment(
    attachment_pairs: list[tuple[str, str]] | None = None,
    n_reps: int = 10,
    days: int = 5,
    lm: LMProvider | None = None,
) -> dict:
    """Run the attachment dating simulation for given attachment pairs.

    Returns a dict mapping (p1attach, p2attach) -> list of {cycle, logs} dicts.
    """
    if lm is None:
        from maslow.lm.openai_provider import OpenAIProvider
        lm = OpenAIProvider()

    if attachment_pairs is None:
        attachment_pairs = [
            ("anxious", "avoidant"),
            ("avoidant", "anxious"),
            ("secure", "secure"),
        ]

    results: dict = {}
    for p1attach, p2attach in attachment_pairs:
        key = (p1attach, p2attach)
        results[key] = []
        for rep in range(n_reps):
            print(f"{key} rep {rep}")
            cycle, logs = generate_dating_cycle(
                p1attach, p2attach, lm, days=days
            )
            results[key].append({"cycle": cycle, "logs": logs})

    return results
