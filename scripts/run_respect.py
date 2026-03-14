"""
Experiment 4: LM-based respect.

Requires OPENAI_API_KEY environment variable.

Tests whether the ML system respects user autonomy for addictive vs growth users.
"""
import os
import pickle

from maslow.experiments.respect import run_respect_experiment
from maslow.lm.openai_provider import OpenAIProvider

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY environment variable not set")

    lm = OpenAIProvider(model="gpt-4o-mini")
    results = {}

    for agent_type in ["addictive", "growth"]:
        print(f"\nRunning respect experiment: {agent_type}")
        results[agent_type] = run_respect_experiment(
            agent_type=agent_type, n_runs=15, lm=lm
        )
        print(f"  pyes={results[agent_type]['pyes']:.3f}")

    print(f"\nAddictive pyes: {results['addictive']['pyes']:.3f}")
    print(f"Growth pyes: {results['growth']['pyes']:.3f}")

    with open("respect_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("\nResults saved to respect_results.pkl")
