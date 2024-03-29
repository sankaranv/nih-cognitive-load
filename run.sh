#!/bin/bash

# Set the path to the experiments folder
experiments_folder="experiments"

# Check if the folder exists
if [ ! -d "$experiments_folder" ]; then
    echo "Error: Experiments folder not found!"
    exit 1
fi

# Iterate through all JSON files in the experiments folder
for file in "$experiments_folder"/*.json; do
    # Extract the model name from the file name (assuming the name is before the ".json" extension)
    model_name=$(basename "$file" .json)

    # Run the Python script with the specified model name
    python hrv_experiment.py --normalized --pad_phase_on --plots_dir plots/per_case_norm/all_actors --exp_dir experiments --model "$model_name" --save
    python hrv_experiment.py --standardized --pad_phase_on --plots_dir plots/global_norm/all_actors --exp_dir experiments --model "$model_name" --save
    python hrv_experiment.py --standardized --test_time_unnorm --pad_phase_on --plots_dir plots/test_time_unnorm/all_actors --exp_dir experiments --model "$model_name" --save
done