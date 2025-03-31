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

noise_levels=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8)

for dataset in "PenDigits" "WalkingSittingStanding" "SpokenArabicDigits"; do
    for noise_level in "${noise_levels[@]}"; do
        echo "Running experiment for dataset: $dataset with noise level: $noise_level"
        python3 ./Experiments/noise/run_noise_experiment.py --dataset "$dataset" --noise_level "$noise_level"
    done
    python3 ./misc/plots/noise_plots.py --dataset "$dataset" --noise_levels "${noise_levels[@]}" 
done

deactivate