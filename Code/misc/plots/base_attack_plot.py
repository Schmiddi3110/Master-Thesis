import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser(description="Generate plots for stolen model accuracy under different noise levels.")
parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset")

args = parser.parse_args()


dataset = args.dataset    

df = pd.read_csv(f"TimeSeriesModels/{dataset}/Results/BaseAttacks/Base_Attack.csv")

architectures = ["LSTM", "RNN", "1_Conv_CNN", "2_Conv_CNN", "3_Conv_CNN"]

for architecture in architectures:
    df_filtered = df[df['Model Pairing'].str.endswith(architecture)]

    model_pairings = df_filtered['Model Pairing'].unique()

    rows = (len(model_pairings) // 3) + 1
    fig, axes = plt.subplots(rows, 3, figsize=(18, 6 * rows))
    axes = axes.flatten()

    handles, labels = [], []
    
    for ax, pairing in zip(axes, model_pairings):
        subset = df_filtered[df_filtered['Model Pairing'] == pairing]
        ax.set_title(f'Original: {pairing.split("-")[0]}; Extracted: {pairing.split("-")[1]}', fontsize=12)
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

    for i in range(len(model_pairings), len(axes)):
        fig.delaxes(axes[i])

    fig.legend(handles, labels, loc='lower center', ncol=5, title="Method Name", fontsize=10, bbox_to_anchor=(0.6, 0.1))

    plt.suptitle(f"Comparison of Stolen Model Accuracy for {architecture} on {dataset}", fontsize=16, y=0.95)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    plt.savefig(f'TimeSeriesModels/{dataset}/Results/BaseAttacks/{architecture}_extracted.jpg', bbox_inches='tight')
    plt.close(fig)