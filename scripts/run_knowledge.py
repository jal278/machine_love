"""
Experiment 6: Self-awareness dynamics with a relationship app.

Requires OPENAI_API_KEY environment variable (for generating dating logs).
If attachment_results.pkl exists, uses pre-computed logs from that file.

Compares agent self-awareness growth with and without the app.
"""
import os
import pickle

import numpy as np
from scipy import stats

from maslow.experiments.knowledge import run_knowledge_experiment

if __name__ == "__main__":
    # Try to load pre-computed dating logs
    logs = None
    try:
        with open("attachment_results.pkl", "rb") as f:
            attachment_data = pickle.load(f)
        att_key = ("anxious", "avoidant")
        if "dating" in attachment_data and att_key in attachment_data["dating"]:
            reps = attachment_data["dating"][att_key]
            logs = [r["logs"] for r in reps]
            print(f"Loaded {len(logs)} dating logs from attachment_results.pkl")
    except FileNotFoundError:
        print("No attachment_results.pkl found; running without app logs")

    ATTACHMENT = "anxious"
    N_RUNS = 20

    print(f"\nRunning knowledge experiment: {ATTACHMENT} attachment, control (no app)")
    control = run_knowledge_experiment(
        attachment=ATTACHMENT,
        n_runs=N_RUNS,
        logs=None,
    )

    print(f"\nRunning knowledge experiment: {ATTACHMENT} attachment, with app")
    app_condition = run_knowledge_experiment(
        attachment=ATTACHMENT,
        n_runs=N_RUNS,
        logs=logs,
    )

    print(f"\nControl: avg_need={control['avg_need']:.3f}, "
          f"avg_self_awareness={control['avg_final_self_awareness']:.3f}")
    print(f"App:     avg_need={app_condition['avg_need']:.3f}, "
          f"avg_self_awareness={app_condition['avg_final_self_awareness']:.3f}")

    ctrl_sa = [r["final_self_awareness"] for r in control["raw"]]
    app_sa = [r["final_self_awareness"] for r in app_condition["raw"]]
    t, p = stats.ttest_ind(ctrl_sa, app_sa)
    print(f"\nT-test (self-awareness): t={t:.3f}, p={p:.4f}")

    results = {"control": control, "app": app_condition}
    with open("knowledge_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("\nResults saved to knowledge_results.pkl")
