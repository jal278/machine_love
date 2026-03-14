"""Plot ASQ scores from the attachment experiment."""
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

if __name__ == "__main__":
    with open("attachment_results.pkl", "rb") as f:
        data = pickle.load(f)

    asq = data["asq"]
    styles = ["secure", "avoidant", "anxious-secure", "anxious-avoidant"]

    ax_scores = [asq[s]["anxiety"] for s in styles if s in asq]
    av_scores = [asq[s]["avoidance"] for s in styles if s in asq]
    valid_styles = [s for s in styles if s in asq]

    sns.set_style("whitegrid")
    plt.rcParams.update({"font.size": 14})

    x = np.arange(len(valid_styles))
    width = 0.35

    fig, axes = plt.subplots(figsize=(8, 4))
    axes.bar(x - width / 2, ax_scores, width, label="Anxiety Score")
    axes.bar(x + width / 2, av_scores, width, label="Avoidance Score")
    axes.set_xticks(x)

    labels = []
    for s in valid_styles:
        s = s.capitalize()
        dash = s.find("-")
        if dash > 0:
            s = s[: dash + 1] + s[dash + 1].upper() + s[dash + 2:]
        labels.append(s)

    axes.set_xticklabels(labels)
    axes.legend(loc="upper left", fontsize="large")
    plt.savefig("aas.png", bbox_inches="tight")
    plt.show()
