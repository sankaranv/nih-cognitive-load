import argparse
import os
import pickle
import sys

from models.mcmc_imputer import MCMCImputer
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
    parser.add_argument("--time_interval", type=int, default=5)
    parser.add_argument("--normalized", default=False, action="store_true")
    parser.add_argument("--burn_in", type=int, default=100)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--lag_length", type=int, default=3)
    parser.add_argument("--save_dataset", default=True, action="store_false")
    parser.add_argument("--pad_phase_on", action="store_true")
    parser.add_argument("--base_model", default="BayesianRidge")
    parser.add_argument("--logging_freq", type=int, default=10)
    parser.add_argument("--verbose", default=False, action="store_true")
    args = parser.parse_args()

    # Load dataset
    dataset = make_dataset_from_file(
        data_dir=args.data_dir,
        time_interval=args.time_interval,
        temporal_features=True,
        static_features=False,
        standardize=False,
        pad_phase_on=True if args.pad_phase_on else False,
    )

    print(f"Pad Phase On: {args.pad_phase_on}")
    if args.normalized:
        print("Normalizing dataset per case")
        dataset = normalize_per_case(dataset)

    # Save dataset
    phase_coding = "pad_phase_on" if args.pad_phase_on else "pad_phase_off"
    file_name = (
        "normalized_original_dataset.pkl" if args.normalized else "original_dataset.pkl"
    )
    save_path = os.path.join(args.data_dir, "processed", phase_coding)
    if args.save_dataset:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(os.path.join(save_path, file_name), "wb") as f:
            pickle.dump(dataset, f)

    # Impute data
    imputer = MCMCImputer(base_model=args.base_model, lag_length=args.lag_length)
    imputer.train(dataset, verbose=args.verbose)
    imputer.save(args.model_dir)
    imputed_dataset, mcmc_samples = imputer.impute(
        dataset=dataset,
        burn_in=args.burn_in,
        max_iter=args.max_iter,
        logging_freq=args.logging_freq,
        verbose=args.verbose,
    )

    if args.save_dataset:
        file_name = (
            "normalized_imputed_dataset.pkl"
            if args.normalized
            else "imputed_dataset.pkl"
        )
        pickle.dump(
            imputed_dataset,
            open(os.path.join(save_path, file_name), "wb"),
        )
        pickle.dump(
            mcmc_samples, open(os.path.join(save_path, "mcmc_samples.pkl"), "wb")
        )

    # Plot imputed data
    imputation_line_plot(
        dataset,
        imputed_dataset,
        time_interval=args.time_interval,
    )
