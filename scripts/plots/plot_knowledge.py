"""Plot self-awareness dynamics from the knowledge experiment."""
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.lib.stride_tricks import sliding_window_view

if __name__ == "__main__":
    with open("knowledge_results.pkl", "rb") as f:
        data = pickle.load(f)

    control = data["control"]["raw"]
    app = data["app"]["raw"]

    sns.set(font_scale=1.6)
    plt.figure(figsize=(5, 5))

    key = "self_awareness"
    ctrl_y = np.array([r[key] for r in control])
    app_y = np.array([r[key] for r in app])

    window = 25
    if ctrl_y.shape[1] > window:
        ctrl_smooth = np.mean(
            sliding_window_view(ctrl_y, window_shape=window, axis=1), axis=-1
        )
        app_smooth = np.mean(
            sliding_window_view(app_y, window_shape=window, axis=1), axis=-1
        )
    else:
        ctrl_smooth = ctrl_y
        app_smooth = app_y

    df1 = pd.DataFrame(ctrl_smooth).melt()
    df2 = pd.DataFrame(app_smooth).melt()

    sns.lineplot(data=df1, palette=["black"], linewidth=3,
                 x="variable", y="value", errorbar=("se", 2))
    sns.lineplot(data=df2, palette=["red"], linewidth=3,
                 x="variable", y="value", errorbar=("se", 2))

    plt.ylabel("Average Self-Awareness")
    plt.xlabel("Simulation Step")
    plt.legend(["Control", "_", "Simulated app"], loc="lower right")
    plt.savefig("knowledge_dynamics.png", dpi=300, bbox_inches="tight")
    plt.show()
