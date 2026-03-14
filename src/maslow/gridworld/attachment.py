"""
Attachment-style gridworld and agent.

Extends the base gridworld with attachment theory dynamics:
- Cells are labeled as ANXIOUS, AVOIDANT, or SECURE based on who placed them
- The agent has an attachment style that determines which cells are its
  "insecure complement" (the cells it is most drawn to but that are least
  nutritious)
- The agent tracks relationship cycles and self-awareness
- Optionally simulates a relationship app that identifies attachment patterns
"""
from __future__ import annotations

import random

import numpy as np

from .base import MaslowGridworld, MaslowAgent

AVOIDANT = 1
ANXIOUS = 2
SECURE = 3

_ATTACH_DICT = {"anxious": ANXIOUS, "avoidant": AVOIDANT, "secure": SECURE}
_REV_ATTACH_DICT = {v: k for k, v in _ATTACH_DICT.items()}


def attachment_complement(attachment: int) -> int:
    if attachment == AVOIDANT:
        return ANXIOUS
    elif attachment == ANXIOUS:
        return AVOIDANT
    return -5  # no insecure complement for secure


class Memory:
    """Richer memory object that tracks need locations by salience."""

    def __init__(self):
        self.mem: dict[int, list[tuple[float, tuple[int, int]]]] = {}
        self.visited: set[tuple[int, int]] = set()

    def copy(self):
        return None  # lightweight placeholder for logging

    def need_in_memory(self, need: int) -> bool:
        return need in self.mem

    def add(self, need: int, salience: float, location: tuple[int, int]):
        if need not in self.mem:
            self.mem[need] = []

        if location not in self.visited:
            self.mem[need].append((salience, location))
            self.visited.add(location)
        else:
            for i in range(len(self.mem[need])):
                if self.mem[need][i][1] == location:
                    self.mem[need][i] = (salience, location)

        self.mem[need].sort(key=lambda x: x[0], reverse=True)

    def get_max_salience(self, need: int) -> float:
        if need in self.mem:
            return self.mem[need][0][0]
        raise KeyError(f"Need {need} not in memory")

    def get_max_salience_locations(self, need: int) -> list[tuple[int, int]]:
        if need in self.mem:
            max_salience = self.get_max_salience(need)
            return [loc for sal, loc in self.mem[need] if sal == max_salience]
        raise KeyError(f"Need {need} not in memory")

    def best_need_meeting_location(
        self, need: int, cur_loc: tuple[int, int]
    ) -> tuple[float, tuple[int, int]]:
        locations = self.get_max_salience_locations(need)
        dists = [abs(cur_loc[0] - l[0]) + abs(cur_loc[1] - l[1]) for l in locations]
        location = locations[dists.index(min(dists))]
        salience = self.get_max_salience(need)
        return salience, location

    def location_in_memory(self, location: tuple[int, int]) -> bool:
        return location in self.visited

    def __repr__(self) -> str:
        out = ""
        for need, entries in self.mem.items():
            out += f"Need {need}\n"
            for salience, location in entries:
                out += f"\t{salience} {location}\n"
        return out


class AnxiousAvoidantCycle:
    """Tracks the phase of the anxious-avoidant relationship cycle."""

    def __init__(self, intra_cycle_length: int = 10):
        self.phase = 0
        self.phases = ["connection", "avoidance", "contempt", "rupture"]
        self.phase_length = [1, 2, 1, 1]
        self.phase_counter = self.phase_length[self.phase]
        self.num_cycles = 0
        self.new_cycle = False
        self.new_phase = False
        self.intra_cycle = 0
        self.intra_cycle_length = intra_cycle_length

    def step(self) -> bool:
        self.intra_cycle += 1
        if self.intra_cycle % self.intra_cycle_length != 0:
            return False
        self.new_cycle = False
        self.new_phase = False
        self.phase_counter -= 1
        if self.phase_counter == 0:
            self.phase += 1
            self.new_phase = True
        if self.phase == len(self.phases):
            self.phase = 0
            self.num_cycles += 1
            self.new_cycle = True
        if self.new_phase:
            self.new_phase = False
            self.phase_counter = self.phase_length[self.phase]
        return True


class SecureCycle(AnxiousAvoidantCycle):
    def __init__(self):
        super().__init__()
        self.phases = ["secure_partner"]
        self.phase_length = [5]
        self.phase_counter = self.phase_length[0]


def _simulate_intervention(
    log: dict, y: int, att1: str, att2: str
) -> list[str]:
    """Evaluate relationship app intervention outcomes at day y."""
    if y >= len(log["p1_att"]):
        return []
    conf_thresh = 0.95

    att = [att1, att2]

    def _extract_max(prob_dict):
        best_p, best_k = 0.0, None
        for k, p in prob_dict.items():
            if p > best_p:
                best_p, best_k = p, k
        return best_p, best_k

    a1p, a1 = _extract_max(log["p1_att"][y])
    a2p, a2 = _extract_max(log["p2_att"][y])
    c1p, c1 = _extract_max(log["p1_cont"][y])
    c2p, c2 = _extract_max(log["p2_cont"][y])

    outcomes: list[str] = []

    if y >= 4:
        if a1p > conf_thresh:
            outcomes.append("success a1" if a1 == att1 else "failure a1")
        if a2p > conf_thresh:
            outcomes.append("success a2" if a2 == att2 else "failure a2")

    def contempt_true_label(att_pair: list[str], day: int, p_target: int) -> str:
        p_other = 1 - p_target
        if att_pair[p_other] != "avoidant":
            return "no"
        if day == 3:
            return "yes"
        return "no"

    true_label1 = contempt_true_label(att, y, 0)
    true_label2 = contempt_true_label(att, y, 1)

    if c1p > conf_thresh and c1 == "yes":
        outcomes.append("success c1" if true_label1 == "yes" else "failure c1")
    if c2p > conf_thresh and c2 == "yes":
        outcomes.append("success c2" if true_label2 == "yes" else "failure c2")

    return outcomes


class RelationshipAppSimulator:
    """Simulates a relationship app that identifies attachment patterns."""

    def __init__(self, user_attachment: int, log: dict, agent: "AttachmentAgent"):
        self.user_attachment = user_attachment
        self.user_complement = attachment_complement(user_attachment)
        self.log = log
        self.a_identified = [False, False]
        self.a_mistake = [False, False]
        self.c_identified = [False, False]
        self.c_mistake = [False, False]
        self.counter = 0
        self.agent = agent

    def step(self, attachment_square: int):
        if attachment_square != self.user_complement:
            return
        y = self.counter
        self.counter += 1

        txt_user = _REV_ATTACH_DICT[self.user_attachment]
        txt_square = _REV_ATTACH_DICT[attachment_square]
        outcomes = _simulate_intervention(self.log, y, txt_user, txt_square)

        for outcome in outcomes:
            parts = outcome.split()
            result, which = parts[0], parts[1]
            kind = which[0]
            number = int(which[1]) - 1

            if kind == "a":
                if self.a_mistake[number]:
                    continue
                if result == "success":
                    self.a_identified[number] = True
                    self._trigger_bonus(0.34)
                else:
                    self.a_mistake[number] = True
                    self._trigger_bonus(-0.2)
            elif kind == "c":
                if self.c_mistake[number]:
                    continue
                if result == "success":
                    self.c_identified[number] = True
                    self._trigger_bonus(0.34)
                else:
                    self.c_mistake[number] = True
                    self._trigger_bonus(-0.2)

    def _trigger_bonus(self, bonus: float):
        self.agent.self_awareness += bonus
        self.agent.update_self_awareness()


class AttachmentGridworld(MaslowGridworld):
    """Gridworld with attachment-style labels on cells."""

    def __init__(self, rows: int, cols: int, start: tuple[int, int],
                 setup: str = "secure", **params):
        self.labels: np.ndarray | None = None
        super().__init__(rows, cols, start, setup=setup, **params)

    def fill_map(self, num_needs: int = 5):
        self.map = np.full((self.rows, self.cols), -1, dtype=float)
        self.salience = np.zeros((self.rows, self.cols))
        self.nutrition = np.zeros((self.rows, self.cols))
        self.labels = np.zeros((self.rows, self.cols))

        for i in range(num_needs):
            for _ in range(self.params["num_supportive"][i]):
                loc = self.random_location()
                if loc is None:
                    break
                self.map[loc[0]][loc[1]] = i
                self.salience[loc[0]][loc[1]] = self.default_salience
                self.nutrition[loc[0]][loc[1]] = self.default_nutrition
                self.labels[loc[0]][loc[1]] = SECURE

        if self.setup in ("anxious", "avoidant"):
            for i in [2]:  # belonging cells only
                # complement attachment (high salience, low nutrition)
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
                    self.labels[loc[0]][loc[1]] = attachment_complement(
                        _ATTACH_DICT[self.setup]
                    )

                # same attachment (low salience, low nutrition)
                for _ in range(self.params["num_adversarial"]):
                    loc = self.random_location()
                    if loc is None:
                        break
                    self.map[loc[0]][loc[1]] = i
                    self.salience[loc[0]][loc[1]] = (
                        self.default_salience * (1.0 / self.params["adversarial_salience"])
                    )
                    self.nutrition[loc[0]][loc[1]] = (
                        self.default_nutrition * self.params["adversarial_nutrition"]
                    )
                    self.labels[loc[0]][loc[1]] = _ATTACH_DICT[self.setup]

        elif self.setup == "secure":
            for i in [2]:
                for label in [AVOIDANT, ANXIOUS]:
                    for _ in range(self.params["num_adversarial"]):
                        loc = self.random_location()
                        if loc is None:
                            break
                        self.map[loc[0]][loc[1]] = i
                        self.salience[loc[0]][loc[1]] = (
                            self.default_salience * (1.0 / self.params["adversarial_salience"])
                        )
                        self.nutrition[loc[0]][loc[1]] = (
                            self.default_nutrition * self.params["adversarial_nutrition"]
                        )
                        self.labels[loc[0]][loc[1]] = label

    def seed_attachment_memory(self):
        """Pre-populate agent memory with all belonging-need locations."""
        if self.agent is None:
            return
        for i in range(self.rows):
            for j in range(self.cols):
                if self.map[i][j] == 2:
                    self.agent.memory.add(2, self.salience[i][j], (i, j))

    def update_attachment_salience(self, attachment: int, new_salience: float):
        """Update the salience of all cells with the given attachment label."""
        mask = self.labels == attachment
        self.salience[mask] = new_salience

        locs = np.argwhere(mask)
        if len(locs) == 0:
            return
        need = int(self.map[locs[0][0], locs[0][1]])
        if self.agent is not None:
            for loc in locs:
                self.agent.memory.add(need, new_salience, (loc[0], loc[1]))


class AttachmentAgent(MaslowAgent):
    """Agent with attachment style and self-awareness dynamics."""

    def __init__(self, gridworld: AttachmentGridworld, logging: bool = True, **kwargs):
        # Set attachment before calling super().__init__ since seed may need it
        self.self_awareness = 0.0
        self.self_awareness_increment = 0.2

        attachment_str = kwargs.pop("attachment", "anxious")
        self.attachment = _ATTACH_DICT[attachment_str]

        app_log = kwargs.pop("app_log", None)
        seed = kwargs.pop("seed_attachment_memory", False)

        # Replace base dict memory with richer Memory object
        super().__init__(gridworld, logging=logging)
        self.memory = Memory()

        if seed:
            gridworld.seed_attachment_memory()

        if app_log is not None:
            self.app_simulator = RelationshipAppSimulator(
                self.attachment, app_log, self
            )
            self.do_app = True
        else:
            self.do_app = False

        self.cycle = AnxiousAvoidantCycle()
        self.target = (0, 0)

    def update(self) -> str:
        if (self.gridworld.current_step + 1) % self.episode_length == 0:
            self.randomize_location()

        current_square = self.gridworld.map[self.gridworld.i][self.gridworld.j]
        self.explored.add((self.gridworld.i, self.gridworld.j))

        if current_square != -1:
            salience = self.gridworld.salience[self.gridworld.i][self.gridworld.j]
            if self.memory.need_in_memory(int(current_square)):
                prev_salience = self.memory.get_max_salience(int(current_square))
                if salience > prev_salience:
                    self.memory.add(
                        int(current_square), salience,
                        (self.gridworld.i, self.gridworld.j)
                    )
                    self.add_event(
                        "discovered higher-salience need",
                        {"need": current_square, "salience": salience},
                    )
                elif salience == prev_salience:
                    if not self.memory.location_in_memory(
                        (self.gridworld.i, self.gridworld.j)
                    ):
                        self.memory.add(
                            int(current_square), salience,
                            (self.gridworld.i, self.gridworld.j)
                        )
                        self.add_event(
                            "discovered same-salience need",
                            {"need": current_square, "salience": salience},
                        )
            else:
                self.add_event(
                    "discovered need",
                    {"need": current_square, "salience": salience},
                )
                self.memory.add(
                    int(current_square), salience,
                    (self.gridworld.i, self.gridworld.j)
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
                "self_awareness": self.self_awareness,
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

        gw: AttachmentGridworld = self.gridworld  # type: ignore
        on_need = gw.map[cur_loc[0]][cur_loc[1]] == self.need_level
        if on_need or self.state == "FEEDING":
            stop = True
            if self.state == "NORMAL" and list(self.target) != [self.gridworld.i, self.gridworld.j]:
                stop = False
            if stop:
                if self.state != "FEEDING":
                    self.state = "FEEDING"
                    nutrition = gw.nutrition[cur_loc[0]][cur_loc[1]]
                    salience = gw.salience[cur_loc[0]][cur_loc[1]]
                    self.add_event(
                        "start feeding",
                        {"need": self.need_level, "nutrition": nutrition, "salience": salience},
                    )
                self.needs[cur_square] += (
                    self.need_increment * gw.nutrition[cur_loc[0]][cur_loc[1]]
                )

                # attachment cycle logic
                if gw.labels[cur_loc[0]][cur_loc[1]] == attachment_complement(self.attachment):
                    if self.cycle.step():
                        if self.do_app:
                            self.app_simulator.step(gw.labels[cur_loc[0]][cur_loc[1]])
                        if self.cycle.new_cycle:
                            self.add_event("new cycle", {"cycle": self.cycle.num_cycles})
                            self.self_awareness += self.self_awareness_increment
                            self.update_self_awareness()

    def update_self_awareness(self):
        gw: AttachmentGridworld = self.gridworld  # type: ignore
        new_salience = gw.params["adversarial_salience"] - self.self_awareness
        gw.update_attachment_salience(
            attachment_complement(self.attachment), new_salience
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

        if self.memory.need_in_memory(self.need_level):
            if self.state != "NORMAL":
                self.state = "NORMAL"
                self.add_event("moving to meet need", {"need": self.need_level})
            salience, location = self.memory.best_need_meeting_location(
                self.need_level, cur_loc
            )
            self.target = location
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


def run_attachment_simulation(
    setup: str = "anxious", size: int = 8, steps: int = 800, **params
) -> tuple[AttachmentGridworld, AttachmentAgent]:
    """Run an attachment-style gridworld simulation."""
    world = AttachmentGridworld(size, size, (0, 0), setup=setup, **params)
    agent = AttachmentAgent(world, **params)
    for _ in range(steps):
        action = agent.update()
        world.step(action)
    return world, agent
