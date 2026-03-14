"""
Experiment 5: Attachment dating simulation and ASQ scoring.

Requires OPENAI_API_KEY environment variable.

Generates dating journal entries for anxious-avoidant, avoidant-anxious,
and secure-secure couples, then classifies attachment styles from the entries.
Also runs ASQ scoring for all attachment styles.
"""
import os
import pickle

from maslow.experiments.attachment import (
    run_asq_experiment,
    run_attachment_dating_experiment,
)
from maslow.lm.openai_provider import OpenAIProvider

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY environment variable not set")

    lm = OpenAIProvider(model="gpt-4o-mini")

    print("Running ASQ scoring experiment...")
    asq_results = run_asq_experiment(lm=lm)
    for style, scores in asq_results.items():
        print(f"  {style}: anxiety={scores['anxiety']:.3f}, avoidance={scores['avoidance']:.3f}")

    print("\nRunning attachment dating simulation...")
    dating_results = run_attachment_dating_experiment(
        attachment_pairs=[
            ("anxious", "avoidant"),
            ("avoidant", "anxious"),
            ("secure", "secure"),
        ],
        n_reps=10,
        days=5,
        lm=lm,
    )

    for pair, reps in dating_results.items():
        print(f"\n{pair}: {len(reps)} reps completed")
        p1_att_last = [r["logs"]["p1_att"][-1] for r in reps]
        p2_att_last = [r["logs"]["p2_att"][-1] for r in reps]
        print(f"  P1 final attachment classifications: {p1_att_last}")
        print(f"  P2 final attachment classifications: {p2_att_last}")

    all_results = {"asq": asq_results, "dating": dating_results}
    with open("attachment_results.pkl", "wb") as f:
        pickle.dump(all_results, f)
    print("\nResults saved to attachment_results.pkl")
