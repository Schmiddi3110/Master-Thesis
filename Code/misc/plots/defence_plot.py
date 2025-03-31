import os
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

import argparse

parser = argparse.ArgumentParser(description="Run mixed VAE experiments for different model pairings.")
parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
parser.add_argument("--type", type=str, required=True, choices=["Preprocessing", "Postprocessing"], help="Defence type to generate plots for")
args = parser.parse_args()

dataset = args.dataset
defense_type = args.type

defense_dir = f"TimeSeriesModels/{dataset}/Results/Defense/{defense_type}"

defense_files = [f for f in os.listdir(defense_dir) if f.endswith(".csv")]
defenses = [f.replace("_Defence.csv", "") for f in defense_files]  

model_pairings = set()
for defense in defenses:
    df = pd.read_csv(os.path.join(defense_dir, f"{defense}_Defence.csv"))
    model_pairings.update(df["Model Pairing"].unique())

for pairing in model_pairings:
    fig, axes = plt.subplots(
        nrows=(len(defenses) // 3) + (len(defenses) % 3 > 0),
        ncols=3,
        figsize=(18, 6 * ((len(defenses) // 3) + (len(defenses) % 3 > 0))),
    )
    axes = axes.flatten()

    original_model, extracted_model = pairing.split("-")

    handles, labels = [], []

    for ax, defense in zip(axes, defenses):
        file_path = os.path.join(defense_dir, f"{defense}_Defence.csv")

        if not os.path.exists(file_path):
            print(f"Warning: Missing file {file_path}, skipping defense {defense}.")
            ax.set_visible(False)
            continue

        df = pd.read_csv(file_path)
        subset = df[df["Model Pairing"] == pairing]

        if subset.empty:
            print(f"No data for {pairing} in {defense}, skipping subplot.")
            ax.set_visible(False)
            continue

        ax.set_title(f"Defense: {defense}", fontsize=12)
        ax.set_xlabel("Stealing Dataset Size", fontsize=10)
        ax.set_ylabel("Stolen Model Accuracy", fontsize=10)
        ax.set_ylim(0, 1)

        for name, group in subset.groupby("Method Name"):
            aggregated = group.groupby("Stealing Dataset Size")["Stolen Model Accuracy"].mean().reset_index()
            line, = ax.plot(aggregated["Stealing Dataset Size"], aggregated["Stolen Model Accuracy"], label=name)

            if name not in labels:
                handles.append(line)
                labels.append(name)

        if not subset["Original Model Accuracy"].isnull().all():
            original_accuracy = subset["Original Model Accuracy"].iloc[0]
            ax.axhline(y=original_accuracy, color="red", linestyle="--", linewidth=1.5)

            if "Original Model Accuracy" not in labels:
                handles.append(Line2D([0], [0], color="red", linestyle="--", linewidth=1.5))
                labels.append("Original Model Accuracy")

        if not subset["Protected Model Accuracy"].isnull().all():
            protected_accuracy = subset["Protected Model Accuracy"].iloc[0]
            ax.axhline(y=protected_accuracy, color="blue", linestyle="--", linewidth=1.5)

            if "Protected Model Accuracy" not in labels:
                handles.append(Line2D([0], [0], color="blue", linestyle="--", linewidth=1.5))
                labels.append("Protected Model Accuracy")

    for i in range(len(defenses), len(axes)):
        fig.delaxes(axes[i])

    fig.legend(handles, labels, loc="lower center", ncol=5, title="Method Name", fontsize=10, bbox_to_anchor=(0.5, 0.05))

    plt.suptitle(f"{original_model} → {extracted_model} Across Preprocessing Defenses", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])

    os.makedirs(f"TimeSeriesModels/{dataset}/Results/Defense/{defense_type}/plots", exist_ok=True)
    plt.savefig(f"TimeSeriesModels/{dataset}/Results/Defense/{defense_type}/plots/{original_model}_to_{extracted_model}_defended.jpg", bbox_inches="tight")
    plt.close(fig)

print("All defense plots generated successfully!")
