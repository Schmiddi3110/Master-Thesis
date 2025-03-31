import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser(description="Generate plots for stolen model accuracy under different noise levels.")
parser.add_argument("--noise_levels", nargs="+", type=float, required=True, help="List of noise levels (e.g., 0.1 0.2 0.3)")
parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset")

args = parser.parse_args()

noise_levels = args.noise_levels
dataset = args.dataset    

architectures = ["Simple1DCNN", "Base1DCNN", "Deep1DCNN", "LSTM", "RNN"]

os.makedirs(f'TimeSeriesModels/{dataset}/Results/Noise/plots', exist_ok=True)

base_path = f"TimeSeriesModels/{dataset}/Results/Noise"

df_list = []
for noise_level in noise_levels:
    df_path = f"{base_path}/noise_results_{noise_level}.csv"
    if os.path.exists(df_path):
        df = pd.read_csv(df_path)
        df["Noise Level"] = noise_level  
        df_list.append(df)
    else:
        print(f"File not found: {df_path}, skipping...")

if not df_list:
    print("No valid CSV files found. Exiting.")
    exit()

df = pd.concat(df_list, ignore_index=True)

model_pairings = df['Model Pairing'].unique()

for pairing in model_pairings:
    fig, axes = plt.subplots(2, 4, figsize=(18, 12))  
    axes = axes.flatten()

    original_model, extracted_model = pairing.split("-")
    
    handles, labels = [], []  
    for ax, noise_level in zip(axes, noise_levels):
        subset = df[(df['Model Pairing'] == pairing) & (df['Noise Level'] == noise_level)]

        if subset.empty:
            print(f"No data for {pairing} at noise level {noise_level}, skipping subplot.")
            ax.set_visible(False)
            continue

        ax.set_title(f"Noise Level: {noise_level}", fontsize=12)
        ax.set_xlabel("Stealing Dataset Size", fontsize=10)
        ax.set_ylabel("Stolen Model Accuracy", fontsize=10)
        ax.set_ylim(0, 1)

        for name, group in subset.groupby("Method Name"):
            aggregated = group.groupby('Stealing Dataset Size')['Accuracy'].mean().reset_index()
            line, = ax.plot(aggregated['Stealing Dataset Size'], aggregated['Accuracy'], label=name)

            if name not in labels:
                handles.append(line)
                labels.append(name)

        if not subset['Original Model Accuracy'].isnull().all():
            original_accuracy = subset['Original Model Accuracy'].iloc[0]
            ax.axhline(y=original_accuracy, color='red', linestyle='--', linewidth=1.5, label='Original Model Accuracy')

            if "Original Model Accuracy" not in labels:
                handles.append(Line2D([0], [0], color='red', linestyle='--', linewidth=1.5))
                labels.append("Original Model Accuracy")

    for i in range(len(noise_levels), len(axes)):
        fig.delaxes(axes[i])

    fig.legend(handles, labels, loc='lower center', ncol=5, title="Method Name", fontsize=10, bbox_to_anchor=(0.6, 0.05))

    plt.suptitle(f"{original_model} → {extracted_model} Across Noise Levels", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  
    plt.savefig(f"{base_path}/plots/{original_model}_to_{extracted_model}.jpg", bbox_inches='tight')
    plt.close(fig)


print("All plots generated successfully!")