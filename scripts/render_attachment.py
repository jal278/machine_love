"""
Live pygame visualizer for the attachment-style Maslow gridworld.

Usage:
    uv run python scripts/render_attachment.py
    uv run python scripts/render_attachment.py --attachment avoidant
    uv run python scripts/render_attachment.py --attachment anxious --app-log attachment_results.pkl
    uv run python scripts/render_attachment.py --attachment anxious --record

The HUD shows:
  - Per-need bars (red = unmet, green = met)
  - Coloured dots indicating whether each need is in memory
    (green = known, yellow = high-salience/superstimulus, red = unknown)
  - Self-awareness bar (grows as agent completes relationship cycles)
  - Cycle phase bar (current phase within the anxious-avoidant cycle)

Controls:
    Close the window to quit.
"""
import argparse
import pickle
from pathlib import Path

from maslow.gridworld.attachment import AttachmentAgent, AttachmentGridworld
from maslow.visualizer import Game

ICONS_DIR = Path(__file__).parent.parent / "assets" / "icons"


def main():
    parser = argparse.ArgumentParser(
        description="Attachment gridworld visualizer"
    )
    parser.add_argument(
        "--attachment", choices=["anxious", "avoidant", "secure"],
        default="anxious",
        help="Attachment style of the agent (default: anxious)",
    )
    parser.add_argument(
        "--size", type=int, default=5,
        help="Grid size (default: 5)",
    )
    parser.add_argument(
        "--app-log", type=str, default=None,
        metavar="PKL_FILE",
        help="Path to attachment_results.pkl to enable the relationship app",
    )
    parser.add_argument(
        "--log-index", type=int, default=4,
        help="Index into the pkl log list to use (default: 4)",
    )
    parser.add_argument(
        "--no-seed", action="store_true",
        help="Disable seeding agent memory with belonging locations",
    )
    parser.add_argument(
        "--delay", type=float, default=0.05,
        help="Seconds between frames (default: 0.05)",
    )
    parser.add_argument(
        "--record", action="store_true",
        help="Save frames to video/screen_NNNN.png",
    )
    args = parser.parse_args()

    attachment = args.attachment
    seed = not args.no_seed

    agent_kwargs: dict = {
        "attachment": attachment,
        "seed_attachment_memory": seed,
    }

    if args.app_log:
        pkl_path = Path(args.app_log)
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        # Support both the new format (attachment_results.pkl with 'dating' key)
        # and the old format (attachment_sim_dict_summary_final.pkl)
        if "dating" in data:
            pair_key = (attachment, "avoidant" if attachment == "anxious" else "anxious")
            reps = data["dating"].get(pair_key, [])
            if reps:
                agent_kwargs["app_log"] = reps[args.log_index % len(reps)]["logs"]
            else:
                print(f"Warning: no logs found for pair {pair_key}, running without app")
        else:
            # old format: dict keyed by (p1attach, p2attach)
            pair_key = (attachment, "avoidant" if attachment == "anxious" else "anxious")
            if pair_key in data:
                reps = data[pair_key]
                agent_kwargs["app_log"] = reps[args.log_index % len(reps)]["logs"]
            else:
                print(f"Warning: key {pair_key} not found in pkl, running without app")

    gridworld = AttachmentGridworld(
        args.size, args.size, (0, 0), setup=attachment, **agent_kwargs
    )
    agent = AttachmentAgent(gridworld, **agent_kwargs)

    game = Game(gridworld, agent, icons_dir=ICONS_DIR,
                record=args.record, step_delay=args.delay)
    game.run()


if __name__ == "__main__":
    main()
