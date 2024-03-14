from utils.data import load_dataset, binary_classification_dataset
from models.joint_linear import JointLinearRegressor
from models.torch_model import JointNNRegressor
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
    model_type = "JointNNClassifier"
elif model_config["base_model"] in ["LSTMBinary"]:
    model_type = "JointLinearRegressor"
elif model_config["base_model"] == "ParameterFreeAutoregressiveModel":
    model_type = "ParameterFreeAutoregressiveModel"
elif model_config["base_model"] == "ParameterFreeAutoregressiveClassifier":
    model_type = "ParameterFreeAutoregressiveClassifier"
else:
    model_type = "JointLinearClassifier"

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
        dataset[case_idx][param_idx, 0, -1, :] = np.array(
            [j for j in range(num_samples)]
        )

# Create binary classification dataset
classification_dataset = binary_classification_dataset(dataset)

# Train model using cross-validation and get trace
model, trace = cross_validation(
    model_type,
    model_config,
    classification_dataset,
    num_folds=5,
    verbose=True,
)

# For each case, print number of 0 and 1 in real dataset, number of 0 and 1 in predicted dataset
# As well as TPR, FPR, TNR, FNR, Accuracy
for param_name in trace:
    for actor_idx, actor_name in enumerate(config.role_names):
        print(f" \n{actor_name}, {param_name}")
        print(f" ------------------- \n")
        for case_idx, case_data in classification_dataset.items():
            case_predictions = trace[param_name]["predictions"][case_idx]
            actor_color = f"C{config.role_colors[actor_name]}"
            param_idx = config.param_indices[param_name]
            imputed_hrv = case_data[param_idx][actor_idx, 0, :]
            num_timesteps = imputed_hrv.shape[-1]

            # Get predicted HRV
            if len(case_predictions.shape) > 1:
                pred_hrv = case_predictions[:, actor_idx]
            else:
                pred_hrv = case_predictions

            # Cut off first seq_len elements from true data
            true_hrv = imputed_hrv[args.seq_len :]

            # Get number of 0 and 1 in real dataset
            num_0_real = np.sum(true_hrv == 0)
            num_1_real = np.sum(true_hrv == 1)

            # Get number of 0 and 1 in predicted dataset
            num_0_pred = np.sum(pred_hrv == 0)
            num_1_pred = np.sum(pred_hrv == 1)

            # Get TPR, FPR, TNR, FNR, Accuracy
            TP = np.sum((true_hrv == 1) & (pred_hrv == 1))
            FP = np.sum((true_hrv == 0) & (pred_hrv == 1))
            TN = np.sum((true_hrv == 0) & (pred_hrv == 0))
            FN = np.sum((true_hrv == 1) & (pred_hrv == 0))
            TPR = TP / (TP + FN)
            FPR = FP / (FP + TN)
            TNR = TN / (TN + FP)
            FNR = FN / (TP + FN)
            Accuracy = (TP + TN) / (TP + TN + FP + FN)

            print(
                f"Case {case_idx:02d} => Ground truth: ({num_0_real:02d}, {num_1_real:02d}) Predicted: ({num_0_pred:02d}, {num_1_pred:02d}) TPR: {TPR:.2f}, FPR: {FPR:.2f}, TNR: {TNR:.2f}, FNR: {FNR:.2f}, Accuracy: {Accuracy:.2f}"
            )

# Plot ROC curves
if args.plot:
    plot_roc_curve(
        model.model_name, trace, classification_dataset, model.seq_len, args.plots_dir
    )
