from visualization.torch_plots import *
from models import *
from utils.create_batches import create_torch_loader_from_dataset
from utils.training import train, val, test, predict

param_names = ["PNS index", "SNS index", "Mean RR", "RMSSD", "LF-HF"]


def run_all_mlp_experiments(
    train_dataset,
    val_dataset,
    test_dataset,
    unimputed_test_dataset,
    plots_dir,
    model_dir,
    device,
    batch_size,
    num_epochs,
    seq_len,
    pred_len,
    learning_rate,
):
    architectures = [
        [50, 100],
        # [100, 200],
        # [150, 400],
        # [128, 256, 128],
        # [64, 150, 64],
        # [50, 100, 100, 50],
        # [100, 200, 200, 100],
        # [150, 400, 400, 150],
    ]
    for hidden_dims in architectures:
        print(f"Running experiment for MLP: {hidden_dims}")
        predictions_dict = mlp_experiment(
            hidden_dims,
            train_dataset,
            val_dataset,
            test_dataset,
            unimputed_test_dataset,
            plots_dir,
            model_dir,
            device,
            batch_size,
            num_epochs,
            seq_len,
            pred_len,
            learning_rate,
        )
        return predictions_dict


def mlp_experiment(
    hidden_dims,
    train_dataset,
    val_dataset,
    test_dataset,
    unimputed_test_dataset,
    plots_dir,
    model_dir,
    device,
    batch_size,
    num_epochs,
    seq_len,
    pred_len,
    learning_rate,
):
    key = list(train_dataset.keys())[0]
    num_features = train_dataset[key].shape[2]
    num_actors = train_dataset[key].shape[1]

    trace = {"train_loss": {}, "val_loss": {}, "test_loss": {}, "predictions": {}}

    predictions_dict = {}

    model_name = "MLP_" + "_".join(map(str, hidden_dims))
    print(f"Training {model_name} model")

    for param in param_names:
        # Create torch dataloaders
        train_loader = create_torch_loader_from_dataset(
            train_dataset,
            seq_len,
            pred_len,
            param,
            batch_size,
            shuffle=True,
        )
        val_loader = create_torch_loader_from_dataset(
            val_dataset,
            seq_len,
            pred_len,
            param,
            batch_size,
            shuffle=False,
        )
        test_loader = create_torch_loader_from_dataset(
            test_dataset,
            seq_len,
            pred_len,
            param,
            batch_size,
            shuffle=False,
        )

        # Set up model for each HRV param by making a copy of the base model
        model = MLP(
            hidden_dims,
            batch_size,
            num_actors,
            seq_len,
            pred_len,
            num_features,
        )

        # Set criterion and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        model.to(device)
        trace["train_loss"][param] = []
        trace["val_loss"][param] = []
        for epoch in range(num_epochs):
            train_loss = train(model, train_loader, optimizer, criterion, device)
            val_loss = val(model, val_loader, optimizer, criterion, device)
            print(f"Epoch {epoch + 1}: Train Loss: {train_loss}, Val Loss: {val_loss}")
            trace["train_loss"][param].append(train_loss)
            trace["val_loss"][param].append(val_loss)

        # Test model
        test_loss = test(model, test_loader, optimizer, criterion, device)
        print(f"{param} Test Loss: {test_loss}")
        trace["test_loss"][param] = test_loss

        # Plot loss curves
        plot_loss_curves(
            trace["train_loss"][param],
            trace["val_loss"][param],
            round(test_loss, 5),
            f"{plots_dir}/5min/loss_curves/{model_name}",
            param,
            model_name,
            hidden_dims,
        )

        # Save model
        if not os.path.exists(f"./{model_dir}/{model_name}/"):
            os.makedirs(f"./{model_dir}/{model_name}/")
        torch.save(
            model.state_dict(),
            f"./{model_dir}/{model_name}/{param}.pt",
        )

        # Load model
        model.load_state_dict(torch.load(f"./{model_dir}/{model_name}/{param}.pt"))
        model.eval()

        # Get predictions
        predictions = predict(model, test_loader, device)

        # Split predictions by case
        case_ids = list(test_dataset.keys())
        case_predictions = {}
        num_samples = 0
        idx = 0
        total = 0
        for case_id in case_ids:
            num_samples = test_dataset[case_id].shape[-1]
            offset = seq_len + pred_len
            case_predictions[case_id] = predictions[idx : idx + num_samples - offset]
            idx += num_samples - offset
            total += num_samples
        trace["predictions"][param] = case_predictions

    return trace
