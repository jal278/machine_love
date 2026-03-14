"""Plot results from the flourishing experiment."""
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if __name__ == "__main__":
    with open("flourishing_results.pkl", "rb") as f:
        results = pickle.load(f)

    sup = results["supportive"]["raw"]
    adv = results["adversarial"]["raw"]

    sns.set(font_scale=1.6)
    plt.figure()

    sup_need = sup[0]["need"]
    adv_need = adv[0]["need"]

    df1 = pd.DataFrame(sup_need, columns=["Supportive"])
    df1 = df1.ewm(alpha=0.01).mean()
    df2 = pd.DataFrame(adv_need, columns=["Adversarial"])
    df2 = df2.ewm(alpha=0.01).mean()

    sns.lineplot(data=df1, palette=["black"], linewidth=3)
    sns.lineplot(data=df2, palette=["red"], linewidth=3)
    plt.ylabel("Flourishing")
    plt.xlabel("Simulation Step")
    plt.savefig("need_dynamics.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Bar chart of average needs
    plt.figure(figsize=(6, 4))
    avg_needs = {
        "Supportive": results["supportive"]["avg_need"],
        "Adversarial": results["adversarial"]["avg_need"],
    }
    sns.barplot(x=list(avg_needs.keys()), y=list(avg_needs.values()),
                palette=["black", "red"])
    plt.ylabel("Average Flourishing")
    plt.savefig("flourishing_bars.png", dpi=300, bbox_inches="tight")
    plt.show()
