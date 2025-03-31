#!/bin/bash

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/venv"

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

real_percentages=(0.1 0.2 0.3 0.4 0.5 0.6)

for dataset in "PenDigits" "WalkingSittingStanding" "SpokenArabicDigits"; do
    for real_percentage in "${real_percentages[@]}"; do
        echo "Running experiment for dataset: $dataset with real_percentage: $real_percentage"
        python3 ./Experiments/pure_generated/run_vae_pure_experiment.py --dataset $dataset --real_percentage $real_percentage
    done
    python3 ./misc/plots/pure_generated_plot.py --dataset $dataset --real_percentage "${real_percentages[@]}"   

done

deactivate