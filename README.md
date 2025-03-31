# Master Thesis

## Overview

This project provides scripts for executing the experiments described in the written part of my thesis. The experiments are written in python. Bash scripts are provided in the `/Code` directory and require execution permissions before running.

## Prerequisites

Ensure you have the following installed:

- Linux/macOS terminal (or WSL for Windows users)
- Bash shell
- cuda (configure used GPU in `/Code/.env` default is cuda:0)


## Setting Up Execution Permissions

Before running any shell script, you need to grant execution permissions. To do this, use the following command:

```bash
chmod +x <script_name>.sh
```

Replace `<script_name>` with the actual name of the script you want to execute.

## Running the Scripts

Once the execution permission is granted, run the script using:

```bash
./<script_name>.sh
```

Example:

```bash
chmod +x setup.sh
./setup.sh
```

The scripts can be run independently from one another. Except `run_noise_generated_defended.sh`, this required `run_defence_experiment.sh` to be run first as it requires the hyperparameters from that experiment.

To run the hyperparamter search, use `run_defence_experiment.sh N` with N being the amount of trials to be run.
