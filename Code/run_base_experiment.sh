#!/bin/bash
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

REQUIREMENTS_FILE="$PROJECT_ROOT/Code/requirements.txt"
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install --upgrade pip
    pip install -r "$REQUIREMENTS_FILE"
else
    echo "Warning: requirements.txt not found in $PROJECT_ROOT. Skipping installation."
fi

for dataset in "PenDigits" "WalkingSittingStanding" "SpokenArabicDigits"; do
    echo "Running experiment for dataset: $dataset"
    python3 ./Experiments/base_attack/run_base_experiment.py --dataset $dataset 
    python3 ./misc/plots/base_attack_plot.py --dataset $dataset
done

deactivate