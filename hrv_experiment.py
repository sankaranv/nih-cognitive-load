from utils.data import (
    load_dataset,
    standardize_dataset,
    rescale_standardized_predictions,
)
from models.joint_linear import JointLinearModel
from models.torch_model import JointNNModel
from utils.training import cross_validation
from visualization.all_plots import *
import argparse
import json

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--normalized", action="store_true")
parser.add_argument("--standardized", action="store_true")
parser.add_argument("--pad_phase_on", action="store_true")
parser.add_argument("--residual", action="store_true")
parser.add_argument("--seq_len", type=int, default=5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--model", type=str, default="ridge")
args = parser.parse_args()

# Set random seed
seed = args.seed
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# Load model config
with open(f"./experiments/{args.model}.json") as f:
    model_config = json.load(f)
if model_config["base_model"] in ["MLP", "LSTM", "ContinuousTransformer"]:
    model_type = "JointNNModel"
elif model_config["base_model"] == "ParameterFreeAutoregressive":
    model_type = "ParameterFreeAutoregressive"
else:
    model_type = "JointLinear"

# Load dataset
dataset, unimputed_dataset = load_dataset(
    data_dir="./data/processed",
    normalized=args.normalized,
    pad_phase_on=args.pad_phase_on,
)
if args.standardized:
    dataset, dataset_means, dataset_stds = standardize_dataset(dataset)
    unimputed_dataset = standardize_dataset(unimputed_dataset)
if args.residual:
    dataset = residual_dataset(dataset)
    unimputed_dataset = (
        dataset  # Haven't figured out how to deal with residual with NaNs
    )

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

if isinstance(model, JointNNModel):
    print("Plotting loss curves")
    plot_loss_curves(model.model_name, trace, "./plots/5min")

# print("Plotting predictions")
# plot_predictions(
#     model.model_name,
#     trace,
#     dataset,
#     unimputed_dataset,
#     model.seq_len,
#     model.pred_len,
#     "./plots/5min",
# )

print("Plotting correlation densities")
eval_metric_density_plots(model.model_name, trace, dataset, "corr_coef", "./plots/5min")

print("Plotting scatterplots")
generate_scatterplots(
    model, trace, dataset, model.seq_len, model.pred_len, "./plots/5min"
)

if isinstance(model, JointLinearModel) and model.base_model == "Ridge":
    print("Plotting feature importances")
    plot_feature_importances(model, model.seq_len, "./plots/5min")
