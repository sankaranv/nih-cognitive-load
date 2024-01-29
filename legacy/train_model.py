from utils.create_batches import create_data_pairs
from utils.data import *
import argparse
from sklearn.linear_model import Ridge
from visualization.numpy_plots import plot_predictions, plot_feature_importances

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="Ridge",
        help="Type of model to train",
    )
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--model_dir", type=str, default="./checkpoints")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--plots_dir", type=str, default="./plots")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq_len", type=int, default=3)
    parser.add_argument("--pred_len", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--criterion", type=str, default="MSELoss")
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--normalized", type=bool, default=True)
    parser.add_argument("--pad_phase_on", type=bool, default=True)
    parser.add_argument("--residual", type=bool, default=False)
    args = parser.parse_args()

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load data
    dataset, unimputed_dataset = load_dataset(
        normalized=args.normalized, pad_phase_on=args.pad_phase_on
    )

    if args.residual:
        dataset = residual_dataset(dataset)
        unimputed_dataset = dataset

    train_dataset, val_dataset, test_dataset = make_train_test_split(dataset)
    num_features = dataset[list(dataset.keys())[0]].shape[-2]

    # Get unimputed datasets
    unimputed_test_dataset = {
        case_id: unimputed_dataset[case_id] for case_id in test_dataset.keys()
    }

    predictions = {}
    for case_id, case_data in test_dataset.items():
        predictions[case_id] = {param: [] for param in param_names}

    for param in param_names:
        train_X, train_y = create_data_pairs(
            train_dataset, lag_length=args.seq_len, param=param
        )
        # Set up model
        model = Ridge()
        model.fit(train_X, train_y)

        # Process predictions into case by case dictionary
        param_idx = param_indices[param]
        for case_id, case_data in test_dataset.items():
            for i in range(case_data.shape[-1] - args.seq_len - args.pred_len):
                # Shape of case data is (num_params, num_actors, num_features, num_timesteps)
                # Get input and output vectors
                hrv_input = case_data[param_idx, :, 0, i : i + args.seq_len].reshape(-1)
                temporal_features = case_data[
                    param_idx, 0, 1:, i + args.seq_len
                ].reshape(-1)
                input = np.concatenate((hrv_input, temporal_features)).reshape(1, -1)
                prediction = model.predict(input)[0][0]
                predictions[case_id][param].append(prediction)

        plot_feature_importances(
            model, f"{args.plots_dir}/predictions/ridge", args.seq_len, param
        )

    # Plot predictions
    plot_predictions(
        predictions,
        test_dataset,
        unimputed_test_dataset,
        args.seq_len,
        args.pred_len,
        f"{args.plots_dir}/predictions/ridge",
    )
