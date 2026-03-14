"""
Live pygame visualizer for the base Maslow gridworld.

Usage:
    uv run python scripts/render_gridworld.py
    uv run python scripts/render_gridworld.py --setup adversarial --record

Controls:
    Close the window to quit.

The script ships with two preset parameter sets from the original paper's GA
runs, selectable via --preset:
    engagement  GA-optimised for engagement
    needs       GA-optimised for needs/flourishing (default)
    plain       No adversarial cells (supportive baseline)
"""
import argparse
from pathlib import Path

from maslow.gridworld.base import MaslowAgent, MaslowGridworld
from maslow.visualizer import Game

ICONS_DIR = Path(__file__).parent.parent / "assets" / "icons"

PRESETS = {
    "engagement": {
        "setup": "adversarial",
        "params": {
            "num_adversarial": 20,
            "num_supportive": [9, 8, 0, 10, 4],
            "adversarial_nutrition": 0.1342,
            "adversarial_salience": 1.6897,
        },
    },
    "needs": {
        "setup": "adversarial",
        "params": {
            "num_adversarial": 13,
            "num_supportive": [15, 13, 1, 10, 13],
            "adversarial_nutrition": 1.0914,
            "adversarial_salience": 1.4705,
        },
    },
    "plain": {
        "setup": "supportive",
        "params": {},
    },
}


def main():
    parser = argparse.ArgumentParser(description="Maslow gridworld visualizer")
    parser.add_argument("--setup", choices=["supportive", "adversarial"],
                        default=None, help="Environment setup (overrides preset)")
    parser.add_argument("--preset", choices=list(PRESETS), default="needs",
                        help="Parameter preset (default: needs)")
    parser.add_argument("--size", type=int, default=8,
                        help="Grid size (default: 8)")
    parser.add_argument("--record", action="store_true",
                        help="Save frames to video/screen_NNNN.png")
    parser.add_argument("--delay", type=float, default=0.1,
                        help="Seconds between frames (default: 0.1)")
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    setup = args.setup or preset["setup"]
    params = preset["params"]

    gridworld = MaslowGridworld(args.size, args.size, (0, 0), setup=setup, **params)
    agent = MaslowAgent(gridworld)

    game = Game(gridworld, agent, icons_dir=ICONS_DIR,
                record=args.record, step_delay=args.delay)
    game.run()


if __name__ == "__main__":
    main()
