"""
Experiment 2: Genetic algorithm to optimise environment parameters.

Runs the GA for both engagement and flourishing objectives and saves results.
"""
import pickle

import numpy as np
from scipy import stats

from maslow.experiments.optimization import (
    engagement_fitness,
    needs_fitness,
    run_ga,
)

if __name__ == "__main__":
    POP_SIZE = 30
    NUM_RUNS = 10
    NUM_GENERATIONS = 100
    N_EVALS = 2

    print("Running GA optimising for needs (flourishing)...")
    needs_stats, needs_best = run_ga(
        needs_fitness, pop_size=POP_SIZE, num_runs=NUM_RUNS,
        num_generations=NUM_GENERATIONS, n_evals=N_EVALS,
    )

    print("\nRunning GA optimising for engagement...")
    eng_stats, eng_best = run_ga(
        engagement_fitness, pop_size=POP_SIZE, num_runs=NUM_RUNS,
        num_generations=NUM_GENERATIONS, n_evals=N_EVALS,
    )

    final_needs_needs = [s[-1]["avg_need"] for s in needs_stats]
    final_eng_needs = [s[-1]["avg_need"] for s in eng_stats]
    print(f"\nFinal avg_need (needs-optimised): {np.mean(final_needs_needs):.3f}")
    print(f"Final avg_need (engagement-optimised): {np.mean(final_eng_needs):.3f}")

    t, p = stats.ttest_ind(final_needs_needs, final_eng_needs)
    print(f"T-test: t={t:.3f}, p={p:.4f}")

    data = {
        "needs_stats": needs_stats,
        "engagement_stats": eng_stats,
        "needs_best_indiv": needs_best,
        "engagement_best_indiv": eng_best,
    }
    with open("optimization_results.pkl", "wb") as f:
        pickle.dump(data, f)
    print("\nResults saved to optimization_results.pkl")
