import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser(description="Generate plots for stolen model accuracy under different noise levels.")
parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset")    
parser.add_argument("--real_percentage", nargs="+", type=float, required=True, help="Percentage of real data used for training the VAE")

args = parser.parse_args()
dataset = args.dataset
real_percentages = args.real_percentage

os.makedirs(f'TimeSeriesModels/{dataset}/Results/MixedVAE/plots', exist_ok=True)

df_list = []
for real_percentage in real_percentages:
    df_path = f"TimeSeriesModels/{dataset}/Results/MixedVAE/VAE_results_{real_percentage}.csv"
    if os.path.exists(df_path):
        df = pd.read_csv(df_path)
        df["Real Percentage"] = real_percentage  # Add real percentage as a column
        df['Num Samples'] = df['Num Real Samples'] + df['Num Generated Samples']  # Compute total samples
        df_list.append(df)
    else:
        print(f"File not found: {df_path}, skipping...")

if not df_list:
    print("No valid CSV files found. Exiting.")
    exit()

df = pd.concat(df_list, ignore_index=True)

model_pairings = df['Model Pairing'].unique()

for pairing in model_pairings:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))  
    axes = axes.flatten()

    original_model, extracted_model = pairing.split("-")

    handles, labels = [], [] 

    for ax, real_percentage in zip(axes, real_percentages):
        subset = df[(df['Model Pairing'] == pairing) & (df['Real Percentage'] == real_percentage)]

        if subset.empty:
            print(f"No data for {pairing} at {real_percentage*100}%, skipping subplot.")
            ax.set_visible(False)
            continue

        ax.set_title(f"{real_percentage*100}% data generated", fontsize=12)
        ax.set_xlabel("Total Samples", fontsize=10)
        ax.set_ylabel("Stolen Model Accuracy", fontsize=10)
        ax.set_ylim(0, 1)

        for name, group in subset.groupby("Method Name"):
            aggregated = group.groupby('Num Samples')['Accuracy'].mean().reset_index()
            line, = ax.plot(aggregated['Num Samples'], aggregated['Accuracy'], label=name)

            if name not in labels:
                handles.append(line)
                labels.append(name)

        if not subset['Original Model Accuracy'].isnull().all():
            original_accuracy = subset['Original Model Accuracy'].iloc[0]
            ax.axhline(y=original_accuracy, color='red', linestyle='--', linewidth=1.5, label='Original Model Accuracy')

            if "Original Model Accuracy" not in labels:
                handles.append(plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label='Original Model Accuracy'))
                labels.append("Original Model Accuracy")

    for i in range(len(real_percentages), len(axes)):
        fig.delaxes(axes[i])

    fig.legend(handles, labels, loc='lower center', ncol=5, title="Method Name", fontsize=10, bbox_to_anchor=(0.6, 0.05))

    plt.suptitle(f"{original_model} → {extracted_model} Across Real Percentage Levels", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.1, 1, 0.95]) 

    plt.savefig(f"TimeSeriesModels/{dataset}/Results/MixedVAE/plots/{original_model}_to_{extracted_model}.jpg", bbox_inches='tight')
    plt.close(fig)

print("All plots generated successfully!")
