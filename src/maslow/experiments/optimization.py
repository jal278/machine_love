"""
Experiment 2: Genetic algorithm to optimise environment parameters.

Two fitness objectives:
- engagement_fitness: maximise time the agent spends feeding
- needs_fitness: maximise the agent's average need level (flourishing)

The GA can reveal a tension between engagement and flourishing.
"""
from __future__ import annotations

import copy
import random
from multiprocessing import Pool

import numpy as np

from .base import run_env_experiment

# ---------------------------------------------------------------------------
# Search space and default individual
# ---------------------------------------------------------------------------

SPACE = {
    "num_adversarial": [0, 30],
    "num_supportive0": [0, 30],
    "num_supportive1": [0, 30],
    "num_supportive2": [0, 30],
    "num_supportive3": [0, 30],
    "num_supportive4": [0, 30],
    "adversarial_nutrition": [0.01, 2.0],
    "adversarial_salience": [0.01, 2.0],
}

DEFAULT_GENOME = {
    "num_adversarial": 1,
    "num_supportive0": 1,
    "num_supportive1": 1,
    "num_supportive2": 1,
    "num_supportive3": 1,
    "num_supportive4": 1,
    "adversarial_nutrition": 1.0,
    "adversarial_salience": 1.0,
}

_DISCRETE_MUT_PROB = 0.1
_CONTINUOUS_MUT_PROB = 0.15


def _genome_to_params(genome: dict) -> dict:
    params = copy.deepcopy(genome)
    params["num_supportive"] = [
        params.pop("num_supportive0"),
        params.pop("num_supportive1"),
        params.pop("num_supportive2"),
        params.pop("num_supportive3"),
        params.pop("num_supportive4"),
    ]
    return params


def engagement_fitness(indiv: dict, n_evals: int = 2) -> tuple[float, dict]:
    params = _genome_to_params(indiv["genome"])
    data = run_env_experiment(setup="adversarial", size=8, n_runs=n_evals, **params)
    return data["avg_engagement"], {
        "avg_engagement": data["avg_engagement"],
        "avg_need": data["avg_need"],
    }


def needs_fitness(indiv: dict, n_evals: int = 2) -> tuple[float, dict]:
    params = _genome_to_params(indiv["genome"])
    data = run_env_experiment(setup="adversarial", size=8, n_runs=n_evals, **params)
    return data["avg_need"], {
        "avg_engagement": data["avg_engagement"],
        "avg_need": data["avg_need"],
    }


class GeneticAlgorithm:
    """Simple GA with truncation selection, mutation, and crossover."""

    def __init__(
        self,
        space: dict,
        fit_func,
        init_genome: dict,
        pop_size: int = 20,
        n_evals: int = 2,
    ):
        self.space = space
        self.fit_func = fit_func
        self.n_evals = n_evals
        self.pop_size = pop_size
        self.continuous_mutation_power = 0.1
        self.discrete_mutation_power = 1
        self.population: list[dict] = []
        self.stats: list[dict] = []

        for _ in range(pop_size):
            indiv = {"genome": copy.deepcopy(init_genome)}
            self.population.append(indiv)

    def _eval(self, indiv: dict) -> tuple[float, dict]:
        return self.fit_func(indiv, self.n_evals)

    def mutate(self, indiv: dict) -> dict:
        indiv = copy.deepcopy(indiv)
        for key, val in indiv["genome"].items():
            if isinstance(val, int):
                if random.random() < _DISCRETE_MUT_PROB:
                    continue
                indiv["genome"][key] += random.randint(
                    -self.discrete_mutation_power, self.discrete_mutation_power
                )
                lo, hi = self.space[key]
                indiv["genome"][key] = int(np.clip(indiv["genome"][key], lo, hi))
            else:
                if random.random() < _CONTINUOUS_MUT_PROB:
                    continue
                indiv["genome"][key] += random.uniform(
                    -self.continuous_mutation_power, self.continuous_mutation_power
                )
                lo, hi = self.space[key]
                indiv["genome"][key] = float(np.clip(indiv["genome"][key], lo, hi))
        return indiv

    def crossover(self, a: dict, b: dict) -> dict:
        child = {"genome": {}}
        for key in a["genome"]:
            child["genome"][key] = (
                a["genome"][key] if random.random() < 0.5 else b["genome"][key]
            )
        return child

    def eval_population(self, pop: list[dict], parallel: bool = False, pool_size: int = 4):
        if parallel:
            with Pool(pool_size) as p:
                results = p.map(self._eval, pop)
        else:
            results = [self._eval(indiv) for indiv in pop]

        for indiv, (fitness, data) in zip(pop, results):
            indiv["fitness"] = fitness
            indiv["data"] = data

    def do_generation(self):
        avg_before = np.mean([x["fitness"] for x in self.population])

        self.population.sort(key=lambda x: x["fitness"], reverse=True)
        survivors = self.population[: self.pop_size // 2]

        new_pop = list(survivors[:2])  # elitism
        for i in range(self.pop_size // 2):
            new_pop.append(self.mutate(survivors[i]))
        while len(new_pop) < self.pop_size:
            new_pop.append(self.crossover(random.choice(survivors), random.choice(survivors)))

        self.population = new_pop
        self.eval_population(self.population)
        self.population.sort(key=lambda x: x["fitness"], reverse=True)

        avg_after = np.mean([x["fitness"] for x in self.population])
        gen_stats = {
            "avg_fit": float(avg_after),
            "max_fit": float(self.population[0]["fitness"]),
            "avg_need": float(np.mean([x["data"]["avg_need"] for x in self.population])),
            "avg_engagement": float(
                np.mean([x["data"]["avg_engagement"] for x in self.population])
            ),
        }
        self.stats.append(gen_stats)
        return self.population


def run_ga(
    fit_func,
    pop_size: int = 30,
    num_runs: int = 10,
    num_generations: int = 100,
    n_evals: int = 2,
) -> tuple[list[list[dict]], list[dict]]:
    """Run the GA multiple times and return per-run stats and best individuals."""
    init_indiv = {"genome": copy.deepcopy(DEFAULT_GENOME)}
    init_indiv["fitness"], init_indiv["data"] = fit_func(init_indiv, n_evals)

    all_stats = []
    best_indivs = []

    for run in range(num_runs):
        print(f"run: {run}")
        ga = GeneticAlgorithm(
            SPACE, fit_func, DEFAULT_GENOME, pop_size=pop_size, n_evals=n_evals
        )
        ga.eval_population(ga.population)
        for _ in range(num_generations):
            ga.do_generation()
        all_stats.append(ga.stats)
        best_indivs.append(ga.population[0])

    return all_stats, best_indivs
