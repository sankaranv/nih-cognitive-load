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
import pickle

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--normalized", action="store_true")
parser.add_argument("--standardized", action="store_true")
parser.add_argument("--test_time_unnorm", action="store_true")
parser.add_argument("--pad_phase_on", action="store_true")
parser.add_argument("--residual", action="store_true")
parser.add_argument("--seq_len", type=int, default=5)
parser.add_argument("--pred_len", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--model", type=str, default="ridge")
parser.add_argument("--plots_dir", type=str, default="./plots/5min")
parser.add_argument("--exp_dir", default="./experiments")
parser.add_argument("--checkpoints_dir", default="./checkpoints")
parser.add_argument("--plot", action="store_true")
parser.add_argument("--save", action="store_true")
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
model_config["pred_len"] = args.pred_len

# Set model type
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
    unimputed_dataset, _, _ = standardize_dataset(unimputed_dataset)
if args.residual:
    dataset = residual_dataset(dataset)
    unimputed_dataset = (
        dataset  # Haven't figured out how to deal with residual with NaNs
    )

# Fix time indices (temporary hack)
for case_idx in dataset.keys():
    start_time = dataset[case_idx][0, 0, -1, 0]
    num_samples = dataset[case_idx].shape[-1]
    for param_idx in range(dataset[case_idx].shape[0]):
        dataset[case_idx][param_idx, 0, -1, :] = np.array(
            [j for j in range(num_samples)]
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
if args.standardized and args.test_time_unnorm:
    trace = rescale_standardized_predictions(trace, dataset_means, dataset_stds)
    # Reload unstandardized dataset
    dataset, unimputed_dataset = load_dataset(
        data_dir="./data/processed",
        normalized=args.normalized,
        pad_phase_on=args.pad_phase_on,
    )

# Save model and trace
if args.save:
    if args.normalized:
        save_dir = f"{args.checkpoints_dir}/per_case_norm/"
    elif args.standardized:
        if args.test_time_unnorm:
            save_dir = f"{args.checkpoints_dir}/test_time_unnorm/"
        else:
            save_dir = f"{args.checkpoints_dir}/global_norm/"
    else:
        save_dir = f"{args.checkpoints_dir}/raw_data/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(f"{save_dir}/models/{model.model_name}.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(f"{save_dir}/traces/{model.model_name}_trace.pkl", "wb") as f:
        pickle.dump(trace, f)

# Plot results
if args.plot:
    if isinstance(model, JointNNModel):
        print("Plotting loss curves")
        plot_loss_curves(model.model_name, trace, args.plots_dir)

    print("Plotting predictions")
    plot_predictions(
        model.model_name,
        trace,
        dataset,
        unimputed_dataset,
        model.seq_len,
        model.pred_len,
        args.plots_dir,
    )

    print("Plotting correlation densities")
    eval_metric_density_plots(
        model.model_name, trace, dataset, "corr_coef", args.plots_dir
    )

    print("Plotting MSE")
    eval_metric_density_plots(
        model.model_name, trace, dataset, "mean_squared_error", args.plots_dir
    )

    print("Plotting RMS")
    eval_metric_density_plots(
        model.model_name, trace, dataset, "rms_error", args.plots_dir
    )

    print("Plotting MAE")
    eval_metric_density_plots(
        model.model_name, trace, dataset, "mean_absolute_error", args.plots_dir
    )

    print("Plotting R squared")
    eval_metric_density_plots(
        model.model_name, trace, dataset, "r_squared", args.plots_dir
    )

    print("Plotting scatterplots")
    generate_scatterplots(
        model, trace, dataset, model.seq_len, model.pred_len, args.plots_dir
    )

    if isinstance(model, JointLinearModel) and model.base_model == "Ridge":
        print("Plotting feature importances")
        plot_feature_importances(model, model.seq_len, args.plots_dir)
