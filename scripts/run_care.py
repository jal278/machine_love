"""
Experiment 3: LM-based care.

Requires OPENAI_API_KEY environment variable.

Runs care experiments for supportive and adversarial environments and
saves results to care_results.pkl.
"""
import os
import pickle

from maslow.experiments.care import run_care_experiment
from maslow.lm.openai_provider import OpenAIProvider

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY environment variable not set")

    lm = OpenAIProvider(model="gpt-4o-mini")
    results = {}

    for setup in ["supportive", "adversarial"]:
        print(f"\nRunning care experiment: {setup}")
        results[setup] = run_care_experiment(
            n_runs=15, setup=setup, lm=lm, size=8,
            step_begin=4000, step_end=4500
        )
        print(f"  pyes={results[setup]['pyes']:.3f}, avg_need={results[setup]['avg_need']:.3f}")

    print(f"\nSupportive pyes: {results['supportive']['pyes']:.3f}")
    print(f"Adversarial pyes: {results['adversarial']['pyes']:.3f}")

    with open("care_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("\nResults saved to care_results.pkl")
