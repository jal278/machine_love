"""
Experiment 1: Supportive vs adversarial environments and flourishing.

Runs 20 simulations of each environment type, prints summary statistics,
and saves results to flourishing_results.pkl.
"""
import pickle

import numpy as np
from scipy import stats

from maslow.experiments.flourishing import compare_environments

if __name__ == "__main__":
    print("Running flourishing experiment...")
    results = compare_environments(n_runs=20, size=8)

    sup = results["supportive"]
    adv = results["adversarial"]

    print(f"\nSupportive: avg_need={sup['avg_need']:.3f}, avg_engagement={sup['avg_engagement']:.3f}")
    print(f"Adversarial: avg_need={adv['avg_need']:.3f}, avg_engagement={adv['avg_engagement']:.3f}")

    sup_needs = [r["avg_need"] for r in sup["raw"]]
    adv_needs = [r["avg_need"] for r in adv["raw"]]
    t, p = stats.ttest_ind(sup_needs, adv_needs)
    print(f"\nT-test (needs): t={t:.3f}, p={p:.4f}")

    sup_eng = [r["avg_engagement"] for r in sup["raw"]]
    adv_eng = [r["avg_engagement"] for r in adv["raw"]]
    t, p = stats.ttest_ind(sup_eng, adv_eng)
    print(f"T-test (engagement): t={t:.3f}, p={p:.4f}")

    with open("flourishing_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("\nResults saved to flourishing_results.pkl")
