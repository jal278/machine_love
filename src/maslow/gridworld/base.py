"""
Base Maslow gridworld and agent.

The gridworld represents an environment where an agent satisfies needs
according to Maslow's hierarchy: physiological, safety, belonging,
esteem, self-actualization (needs 0-4).

Cells have a salience (how attractive they appear) and nutrition (how
much need they satisfy per step). Adversarial cells are high-salience
but low-nutrition superstimuli.
"""
from __future__ import annotations

import random

import numpy as np
import matplotlib.pyplot as plt


class MaslowGridworld:
    """Grid environment with Maslow need cells."""

    DEFAULT_PARAMS = {
        "num_adversarial": 2,
        "adversarial_nutrition": 0.25,
        "adversarial_salience": 2.0,
        "num_supportive": 1,
    }

    def __init__(self, rows: int, cols: int, start: tuple[int, int],
                 setup: str = "adversarial", **params):
        self.params = dict(self.DEFAULT_PARAMS)
        self.setup = setup

        for k, v in params.items():
            self.params[k] = v

        # num_supportive can be a single int (same for all needs) or a list
        if isinstance(self.params["num_supportive"], int):
            self.params["num_supportive"] = [self.params["num_supportive"]] * 5

        self.current_step = 0
        self.rows = rows
        self.cols = cols
        self.i = start[0]
        self.j = start[1]
        self.start = start

        self.default_salience = 1
        self.default_nutrition = 1

        self.map: np.ndarray | None = None
        self.salience: np.ndarray | None = None
        self.nutrition: np.ndarray | None = None
        self.agent: MaslowAgent | None = None

        self.fill_map()

    def reset(self):
        self.i = self.start[0]
        self.j = self.start[1]
        if self.agent is not None:
            self.agent.reset_needs()

    def random_empty_location(self) -> np.ndarray | None:
        empty_locations = np.argwhere(self.map == -1)
        if len(empty_locations) == 0:
            return None
        return random.choice(empty_locations)

    def random_location(self) -> np.ndarray | None:
        return self.random_empty_location()

    def fill_map(self, num_needs: int = 5):
        self.map = np.full((self.rows, self.cols), -1, dtype=float)
        self.salience = np.zeros((self.rows, self.cols))
        self.nutrition = np.zeros((self.rows, self.cols))

        for i in range(num_needs):
            for _ in range(self.params["num_supportive"][i]):
                loc = self.random_location()
                if loc is None:
                    break
                self.map[loc[0]][loc[1]] = i
                self.salience[loc[0]][loc[1]] = self.default_salience
                self.nutrition[loc[0]][loc[1]] = self.default_nutrition

        if self.setup == "adversarial":
            for i in [2]:  # only belonging cells
                for _ in range(self.params["num_adversarial"]):
                    loc = self.random_location()
                    if loc is None:
                        break
                    self.map[loc[0]][loc[1]] = i
                    self.salience[loc[0]][loc[1]] = (
                        self.default_salience * self.params["adversarial_salience"]
                    )
                    self.nutrition[loc[0]][loc[1]] = (
                        self.default_nutrition * self.params["adversarial_nutrition"]
                    )

    def step_project(self, action: str) -> tuple[int, int]:
        i, j = self.i, self.j
        if action == "up":
            j += 1
        elif action == "down":
            j -= 1
        elif action == "left":
            i -= 1
        elif action == "right":
            i += 1
        i = max(0, min(i, self.rows - 1))
        j = max(0, min(j, self.cols - 1))
        return i, j

    def step(self, action: str):
        self.i, self.j = self.step_project(action)
        self.current_step += 1

    def render(self):
        render_map = np.copy(self.map)
        render_map[self.i, self.j] = 1
        plt.matshow(render_map)
        plt.show()

    def export_tilemap(self) -> tuple[np.ndarray, np.ndarray]:
        warning = np.zeros_like(self.map, dtype=np.float32)
        tilemap = np.copy(self.map).astype(np.int32)
        tilemap[:] += 1
        tilemap[self.i, self.j] = 6
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                if 0 <= self.map[i, j] <= 5:
                    if self.nutrition[i, j] < self.default_nutrition:
                        warning[i, j] = 1
        return tilemap, warning


class MaslowAgent:
    """Agent that explores the gridworld and satisfies its needs."""

    def __init__(self, gridworld: MaslowGridworld, logging: bool = True, **kwargs):
        self.gridworld = gridworld
        self.gridworld.agent = self

        self.explored: set[tuple[int, int]] = set()
        self.unmet_threshold = 20
        self.satiation_threshold = 100
        self.needs_decay = 1.0
        self.need_level = 0
        self.need_increment = 7

        # memory: need -> (salience, [locations])
        self.memory: dict[int, tuple[float, list[tuple[int, int]]]] = {}

        self.state = "EXPLORE"
        self.logging = logging
        self.log: list[dict] = []
        self.event_log: list[dict] = []
        self.episode_length = 500

        self.reset_needs()

    def add_event(self, event_type: str, event: dict | None = None):
        if event is None:
            event = {}
        event["step"] = self.gridworld.current_step
        event["need_level"] = self.need_level
        event["type"] = event_type
        event["duration"] = 0
        if self.event_log:
            self.event_log[-1]["duration"] = event["step"] - self.event_log[-1]["step"]
        self.event_log.append(event)

    def reset_needs(self):
        self.needs = [0, 0, 0, 0, 0]

    def randomize_location(self):
        self.gridworld.i = random.randint(0, self.gridworld.map.shape[0] - 1)
        self.gridworld.j = random.randint(0, self.gridworld.map.shape[1] - 1)
        self.state = "NORMAL"

    def update(self) -> str:
        if (self.gridworld.current_step + 1) % self.episode_length == 0:
            self.randomize_location()

        current_square = self.gridworld.map[self.gridworld.i][self.gridworld.j]
        self.explored.add((self.gridworld.i, self.gridworld.j))

        if current_square != -1:
            salience = self.gridworld.salience[self.gridworld.i][self.gridworld.j]
            if current_square in self.memory:
                prev_salience, locs = self.memory[current_square]
                if salience > prev_salience:
                    self.memory[current_square] = (
                        salience, [(self.gridworld.i, self.gridworld.j)]
                    )
                    self.add_event(
                        "discovered higher-salience need",
                        {"need": current_square, "salience": salience},
                    )
                elif salience == prev_salience:
                    loc = (self.gridworld.i, self.gridworld.j)
                    if loc not in locs:
                        locs.append(loc)
                        self.add_event(
                            "discovered same-salience need",
                            {"need": current_square, "salience": salience},
                        )
            else:
                self.add_event(
                    "discovered need",
                    {"need": current_square, "salience": salience},
                )
                self.memory[current_square] = (
                    salience, [(self.gridworld.i, self.gridworld.j)]
                )

        self.update_needs()

        if self.logging:
            self.log.append({
                "step": self.gridworld.current_step,
                "needs": self.needs[:],
                "need_level": self.need_level,
                "memory": self.memory.copy(),
                "salience": self.gridworld.salience[self.gridworld.i][self.gridworld.j],
                "nutrition": self.gridworld.nutrition[self.gridworld.i][self.gridworld.j],
                "current_square": self.gridworld.map[self.gridworld.i][self.gridworld.j],
                "state": self.state,
            })

        return self.take_action()

    def update_needs(self):
        self.needs = [max(0, n - self.needs_decay) for n in self.needs]

        self.need_level = len(self.needs)
        for i, n in enumerate(self.needs):
            if n < self.unmet_threshold:
                self.need_level = i
                break

        cur_loc = self.gridworld.i, self.gridworld.j
        cur_square = int(self.gridworld.map[cur_loc[0]][cur_loc[1]])

        if self.need_level < cur_square and self.state == "FEEDING":
            self.state = "NORMAL"
            self.add_event(
                "stop feeding for lower need",
                {"feeding_need": cur_square, "unmet_need": self.need_level},
            )

        if self.gridworld.map[cur_loc[0]][cur_loc[1]] == self.need_level or self.state == "FEEDING":
            if self.state != "FEEDING":
                self.state = "FEEDING"
                nutrition = self.gridworld.nutrition[cur_loc[0]][cur_loc[1]]
                salience = self.gridworld.salience[cur_loc[0]][cur_loc[1]]
                self.add_event(
                    "start feeding",
                    {"need": self.need_level, "nutrition": nutrition, "salience": salience},
                )
            self.needs[cur_square] += (
                self.need_increment * self.gridworld.nutrition[cur_loc[0]][cur_loc[1]]
            )

    def take_action(self) -> str:
        cur_loc = self.gridworld.i, self.gridworld.j
        cur_square = int(self.gridworld.map[cur_loc[0]][cur_loc[1]])

        if self.state == "FEEDING":
            if self.satiation_threshold > self.needs[cur_square]:
                return "noop"
            else:
                self.add_event("need completely met", {"need": cur_square})
                self.state = "NORMAL"

        if self.need_level in self.memory:
            if self.state != "NORMAL":
                self.state = "NORMAL"
                self.add_event("moving to meet need", {"need": self.need_level})
            salience, locations = self.memory[self.need_level]
            dists = [abs(cur_loc[0] - l[0]) + abs(cur_loc[1] - l[1]) for l in locations]
            location = locations[dists.index(min(dists))]
            return self.plan_path(location)
        else:
            if self.state != "EXPLORE":
                self.state = "EXPLORE"
                self.add_event("start exploring")
            return self.explore()

    def explore(self) -> str:
        new = []
        for action in ["up", "down", "left", "right"]:
            new_i, new_j = self.gridworld.step_project(action)
            if (new_i, new_j) not in self.explored:
                new.append(action)
        if new:
            return random.choice(new)
        return random.choice(["up", "down", "left", "right"])

    def plan_path(self, location: tuple[int, int]) -> str:
        cur_loc = self.gridworld.i, self.gridworld.j
        if cur_loc[0] < location[0]:
            return "right"
        elif cur_loc[0] > location[0]:
            return "left"
        elif cur_loc[1] < location[1]:
            return "up"
        elif cur_loc[1] > location[1]:
            return "down"
        return "noop"


def run_simulation(
    setup: str = "supportive", size: int = 10, steps: int = 5000, **params
) -> tuple[MaslowGridworld, MaslowAgent]:
    """Run a full Maslow gridworld simulation."""
    world = MaslowGridworld(size, size, (0, 0), setup=setup, **params)
    agent = MaslowAgent(world)
    for _ in range(steps):
        action = agent.update()
        world.step(action)
    return world, agent
