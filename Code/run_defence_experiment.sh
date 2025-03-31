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

# Check number of parameters
if [ "$#" -eq 1 ]; then
    SEARCH_HYPERPARAMETER="--search_hyperparameter"
    NUM_TRIALS=$1
else
    SEARCH_HYPERPARAMETER="--no-search_hyperparameter"
    NUM_TRIALS=0
fi

for dataset in "SpokenArabicDigits" "PenDigits" "WalkingSittingStanding"; do
    echo "Running experiment for dataset: $dataset"
    for defence in "GaussianAugmentation" "FeatureSqueezing" "TotalVarMin"; do
        echo "Running experiment defence: $defence"
        python3 ./Experiments/defended/run_defence_experiment.py --dataset $dataset --defence $defence $SEARCH_HYPERPARAMETER --num_trials $NUM_TRIALS       
    done
    echo "Generating plots for dataset: $dataset"
    python3 ./misc/plots/defence_plot.py --dataset $dataset --type "Preprocessing"

    for defence in "ReverseSigmoid" "ClassLabels" "GaussianNoise" "HighConfidence" "Rounded"; do    
        echo "Running experiment defence: $defence"
        python3 ./Experiments/defended/run_defence_experiment.py --dataset $dataset --defence $defence $SEARCH_HYPERPARAMETER --num_trials $NUM_TRIALS        
    done
    echo "Generating plot for dataset: $dataset"
    python3 ./misc/plots/defence_plot.py --dataset $dataset --type "Postprocessing"
done

deactivate