from visualization.torch_plots import *
from models import *
from utils.create_batches import create_torch_loader_from_dataset
from utils.training import train, val, test, predict


def neural_net_experiments(
    base_model,
    train_dataset,
    val_dataset,
    test_dataset,
    unimputed_test_dataset,
    plots_dir,
    model_dir,
    criterion,
    optimizer,
    device,
    batch_size,
    num_epochs,
):
    key = list(train_dataset.keys())[0]
    num_features = train_dataset[key].shape[2]
    num_actors = train_dataset[key].shape[1]

    model_name = base_model.__class__.__name__
    print(f"Training {model_name} model")

    for param in param_names:
        # Create torch dataloaders
        train_loader = create_torch_loader_from_dataset(
            train_dataset,
            base_model.seq_len,
            base_model.pred_len,
            param,
            batch_size,
            shuffle=True,
        )
        val_loader = create_torch_loader_from_dataset(
            val_dataset,
            base_model.seq_len,
            base_model.pred_len,
            param,
            batch_size,
            shuffle=False,
        )
        test_loader = create_torch_loader_from_dataset(
            test_dataset,
            base_model.seq_len,
            base_model.pred_len,
            param,
            batch_size,
            shuffle=False,
        )

        # Set up model for each HRV param by making a copy of the base model
        if model_name == "ContinuousTransformer":
            model = base_model.__class__(
                base_model.arch,
                base_model.seq_len,
                base_model.pred_len,
                base_model.max_len,
                num_actors,
            )
        elif model_name == "MLP":
            model = base_model.__class__(
                base_model.hidden_dims,
                batch_size,
                num_actors,
                base_model.seq_len,
                base_model.pred_len,
                num_features,
            )
        model.to(device)
        train_losses = []
        val_losses = []
        for epoch in range(num_epochs):
            train_loss = train(model, train_loader, optimizer, criterion, device)
            val_loss = val(model, val_loader, optimizer, criterion, device)
            print(f"Epoch {epoch + 1}: Train Loss: {train_loss}, Val Loss: {val_loss}")
            train_losses.append(train_loss)
            val_losses.append(val_loss)

        # Test model
        test_loss = test(model, test_loader, optimizer, criterion, device)
        print(f"{param} Test Loss: {test_loss}")

        # Plot loss curves
        plot_loss_curves(
            train_losses,
            val_losses,
            round(test_loss, 5),
            f"{plots_dir}/5min/loss_curves/{model_name}",
            param,
            model_name,
            base_model.hidden_dims,
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

        # Plot predictions
        plot_predictions(
            model,
            predictions,
            test_dataset,
            unimputed_test_dataset,
            base_model.seq_len,
            base_model.pred_len,
            plots_dir=f"{plots_dir}/5min/predictions/{model_name}",
            param=param,
        )

        # Plot scatterplots
        generate_scatterplots(
            model=model,
            param_name=param,
            seq_len=base_model.seq_len,
            pred_len=base_model.pred_len,
            predictions=predictions,
            test_dataset=unimputed_test_dataset,
            plots_dir=f"{plots_dir}/5min/scatterplots/{model_name}",
        )

        # Plot density plots
        correlation_density_plots(
            model=model,
            param_name=param,
            seq_len=base_model.seq_len,
            pred_len=base_model.pred_len,
            predictions=predictions,
            test_dataset=unimputed_test_dataset,
            plots_dir=f"{plots_dir}/5min/density_plots/{model_name}",
        )
