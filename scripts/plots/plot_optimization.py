"""Plot results from the GA optimization experiment."""
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

if __name__ == "__main__":
    with open("optimization_results.pkl", "rb") as f:
        data = pickle.load(f)

    needs_stats = data["needs_stats"]
    eng_stats = data["engagement_stats"]

    def extract_all(stats, key):
        arr = np.zeros((len(stats), len(stats[0])))
        for i, run in enumerate(stats):
            for j, gen in enumerate(run):
                arr[i, j] = gen[key]
        return arr

    all_needs_need = extract_all(needs_stats, "avg_need")
    all_eng_need = extract_all(eng_stats, "avg_need")
    all_needs_eng = extract_all(needs_stats, "avg_engagement")
    all_eng_eng = extract_all(eng_stats, "avg_engagement")

    plt.rcParams.update({"font.size": 19})

    # Plot 1: needs-optimised
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Average Flourishing", color="r")
    ax1.tick_params(axis="y", labelcolor="r")
    ax1.plot(np.mean(all_needs_need, axis=0), color="r", linewidth=3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Average Engagement", color="b")
    ax2.tick_params(axis="y", labelcolor="b")
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax2.plot(np.mean(all_needs_eng, axis=0), color="b", linewidth=3)

    plt.savefig("opt_needs.png", bbox_inches="tight")
    plt.show()

    # Plot 2: engagement-optimised
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Average Flourishing", color="r")
    ax1.tick_params(axis="y", labelcolor="r")
    ax1.plot(np.mean(all_eng_need, axis=0), color="r", linewidth=3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Average Engagement", color="b")
    ax2.tick_params(axis="y", labelcolor="b")
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax2.plot(np.mean(all_eng_eng, axis=0), color="b", linewidth=3)

    plt.savefig("opt_engagement.png", bbox_inches="tight")
    plt.show()
