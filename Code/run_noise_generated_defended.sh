#!/bin/bash
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

REQUIREMENTS_FILE="$PROJECT_ROOT/Code/requirements.txt"
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install --upgrade pip
    pip install -r "$REQUIREMENTS_FILE"
else
    echo "Warning: requirements.txt not found in $PROJECT_ROOT. Skipping installation."
fi

for dataset in "SpokenArabicDigits" "PenDigits"; do

    echo "Running experiment for dataset: $dataset"
    for defence in "GaussianAugmentation" "FeatureSqueezing" "TotalVarMin"; do        
        echo "Running experiment defence: $defence"
        python3 ./Experiments/noise_generated_defended/run_noise_generated_defended_experiment.py --dataset $dataset --defence $defence --noise_level 0.1
    done
    python3 ./misc/plots/noise_VAE_defence_plot.py --dataset $dataset --type "Preprocessing"

    for defence in "ReverseSigmoid" "ClassLabels" "GaussianNoise" "HighConfidence" "Rounded"; do    
        echo "Running experiment defence: $defence"
        python3 ./Experiments/noise_generated_defended/run_noise_generated_defended_experiment.py --dataset $dataset --defence $defence --noise_level 0.1
    done
    echo "Completed experiments for dataset: $dataset"
    python3 ./misc/plots/noise_VAE_defence_plot.py --dataset $dataset --type "Postprocessing"
done

deactivate