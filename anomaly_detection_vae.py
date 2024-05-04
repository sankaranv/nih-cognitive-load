import argparse
import os
import pickle
import sys
import numpy as np
import random
import torch
import json
from models.vae import VAEAnomalyDetector
from utils.data import load_dataset, discretize_by_percentiles
from visualization.all_plots import *
from utils.data import (
    make_dataset_from_file,
    normalize_per_case,
    prune_phases,
    prune_actors,
)
from utils.config import config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--plots_dir", type=str, default="./plots")
    parser.add_argument("--model_dir", type=str, default="./checkpoints")
    parser.add_argument("--exp_dir", default="./experiments/anomaly")
    parser.add_argument("--time_interval", type=int, default=5)
    parser.add_argument("--normalized", default=True, action="store_true")
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--save_dataset", default=True, action="store_false")
    parser.add_argument("--pad_phase_on", action="store_true")
    parser.add_argument("--model", default="vae")
    parser.add_argument("--logging_freq", type=int, default=10)
    parser.add_argument("--verbose", default=False, action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Set random seed
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load model config
    with open(f"{args.exp_dir}/{args.model}.json") as f:
        model_config = json.load(f)
    model_config["seq_len"] = args.seq_len

    # Load dataset
    dataset, unimputed_dataset = load_dataset(
        data_dir="./data/processed",
        normalized=args.normalized,
        pad_phase_on=args.pad_phase_on,
    )

    # Fix time indices (temporary hack)
    for case_idx in dataset.keys():
        start_time = dataset[case_idx][0, 0, -1, 0]
        num_samples = dataset[case_idx].shape[-1]
        for param_idx in range(dataset[case_idx].shape[0]):
            dataset[case_idx][param_idx, :, -1, :] = np.array(
                [j for j in range(num_samples)]
            )
            unimputed_dataset[case_idx][param_idx, :, -1, :] = np.array(
                [j for j in range(num_samples)]
            )

    # Train VAE
    vae = VAEAnomalyDetector(model_config)
    trace = vae.train(dataset, verbose=False)

    # Save model
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    pickle.dump(vae, open(f"{args.model_dir}/{model_config['base_model']}.pkl", "wb"))

    # Predict probabilities
    probs = vae.predict_proba(dataset, trace=trace)

    # Dump the probabilities to file
    model_name = model_config["base_model"]
    if not os.path.exists(f"{args.model_dir}/anomaly"):
        os.makedirs(f"{args.model_dir}/anomaly")
    with open(f"{args.model_dir}/anomaly/{model_name}_probs.pkl", "wb") as f:
        pickle.dump(probs, f)

    # Plot the dataset with predicted probabilities as colors
    with open(f"{args.model_dir}/anomaly/{model_name}_probs.pkl", "rb") as f:
        probs = pickle.load(f)
    print("Loaded probs")
    plot_anomalies(
        dataset,
        probs,
        model_config["base_model"],
        args.seq_len,
        f"{args.plots_dir}/{model_name}",
    )
