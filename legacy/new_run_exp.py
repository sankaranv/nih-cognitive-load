import sys

from experiments import *
from utils.globals import *

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
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--pred_len", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--criterion", type=str, default="MSELoss")
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--normalized", type=bool, default=True)
    parser.add_argument("--pad_phase_on", type=bool, default=True)
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

    # Prune phases and actors
    dataset = prune_phases(dataset)
    dataset = prune_actors(dataset, actors_to_keep=["Surg"])

    # # Residualize data
    # dataset = residual_dataset(dataset)
    # unimputed_dataset = dataset

    num_features = dataset[list(dataset.keys())[0]].shape[-2]

    # Set up model
    hidden_dims = [50, 100]
    num_folds = 5
    model = MLP(
        hidden_dims,
        args.batch_size,
        num_actors,
        args.seq_len,
        args.pred_len,
        num_features,
    )

    # Model name
    model_name = "MLP_" + "_".join(map(str, hidden_dims))

    # Get predictions by cross validation
    predictions = cross_validate_experiment(
        model=model,
        model_name=model_name,
        dataset=dataset,
        unimputed_dataset=unimputed_dataset,
        plots_dir=args.plots_dir,
        model_dir=args.model_dir,
        num_folds=num_folds,
        hidden_dims=hidden_dims,
        device=device,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        learning_rate=args.lr,
    )

    # Make plots
    for param in param_names:
        # Plot predictions
        plot_predictions(
            model,
            predictions[param],
            dataset,
            unimputed_dataset,
            args.seq_len,
            args.pred_len,
            plots_dir=f"{args.plots_dir}/5min/predictions/{model_name}",
            param=param,
        )

        # Plot scatterplots
        # generate_scatterplots(
        #     model=model,
        #     param_name=param,
        #     seq_len=args.seq_len,
        #     pred_len=args.pred_len,
        #     predictions=predictions[param],
        #     test_dataset=unimputed_dataset,
        #     plots_dir=f"{args.plots_dir}/5min/scatterplots/{model_name}",
        # )

        # Plot densities of correlation coefficients
        torch_correlation_density_plots(
            model_name=model_name,
            param_name=param,
            predictions=predictions,
            test_dataset=dataset,
            plots_dir=f"{args.plots_dir}/5min/density_plots/{model_name}",
        )
