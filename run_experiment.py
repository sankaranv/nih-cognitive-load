from utils.data import load_dataset, standardize_dataset, rescale_standardized_predictions
from utils.training import cross_validation
from visualization.all_plots import *
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="MLP",
    help="Type of model to train",
)
parser.add_argument("--normalized", action="store_true")
parser.add_argument("--standardized", action="store_true")
parser.add_argument("--pad_phase_on", action="store_true")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# Set random seed
seed = args.seed
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# Create model config
if args.model == "JointLinear":
    model_type = "JointLinear"
    model_config = {
        "model_name": "JointLinear",
        "seq_len": 5,
        "pred_len": 1,
    }
elif args.model == "ParamFreeAutoregressive":
    model_type = "ParameterFreeAutoregressive"
    model_config = {
        "model_name": "ParamFreeAutoregressive",
    }
elif args.model == "MLP":
    model_type = "JointNNModel"
    model_config = {
        "model_name": "MLP (128,64)",
        "seq_len": 5,
        "pred_len": 1,
        "lr": 0.001,
        "batch_size": 32,
        "num_epochs": 20,
        "num_features": len(config.phases) + 2,
        "hidden_dims": [128, 64],
        "dropout": 0.1,
    }
else:
    raise ValueError(f"Invalid model type: {args.model}")

# Load dataset
dataset, unimputed_dataset = load_dataset(
    data_dir="./data/processed", normalized=args.normalized, pad_phase_on=args.pad_phase_on
)
if args.standardized:
    dataset, dataset_means, dataset_stds = standardize_dataset(dataset)
    unimputed_dataset = standardize_dataset(unimputed_dataset)

# Train model using cross-validation and get trace
model, trace = cross_validation(
    model_type,
    model_config,
    dataset,
    num_folds=5,
    verbose=True,
)

# Rescale standardized predictions
if args.standardized:
    trace = rescale_standardized_predictions(trace, dataset_means, dataset_stds)

print("Plotting loss curves")
plot_loss_curves(model.model_name, trace, "./plots/5min")

print("Plotting predictions")
plot_predictions(
    model.model_name,
    trace,
    dataset,
    unimputed_dataset,
    model.seq_len,
    model.pred_len,
    "./plots/5min",
)

print("Plotting correlation densities")
eval_metric_density_plots(model.model_name, trace, dataset, "corr_coef", "./plots/5min")

print("Plotting scatterplots")
generate_scatterplots(model, trace, dataset, model.seq_len, model.pred_len, "./plots/5min")

print("Plotting feature importances")
plot_feature_importances(model, model.seq_len, "./plots/5min")
