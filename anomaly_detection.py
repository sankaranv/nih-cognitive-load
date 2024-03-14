import argparse
import os
import pickle
import sys
import numpy as np
import random
import torch
import json
from models.dependency_network import DependencyNetwork
from utils.data import load_dataset
from visualization.imputation_plots import imputation_line_plot
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
    parser.add_argument("--model", default="random_forest")
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
    with open(f"./{args.exp_dir}/{args.model}.json") as f:
        model_config = json.load(f)
    model_config["seq_len"] = args.seq_len

    # Load dataset
    dataset, unimputed_dataset = load_dataset(
        data_dir="./data/processed",
        normalized=args.normalized,
        pad_phase_on=args.pad_phase_on,
    )

    # Predict probabilities for each point in the dataset
    dependency_network = DependencyNetwork(model_config)
    dependency_network.train(dataset)
    dependency_network.save(args.model_dir)
    probs = dependency_network.predict_proba(
        dataset, model_config["burn_in"], model_config["max_iter"]
    )
