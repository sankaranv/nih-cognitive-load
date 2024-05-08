import os.path
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
import random
from utils.create_batches import make_train_test_split, HRVDataset
from models.transformer import ContinuousTransformer
import sys
import matplotlib.pyplot as plt
from visualization.torch_plots import plot_predictions, generate_scatterplots


def predict(model, test_loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model.forward(data)
            predictions.append(output)
    predictions = torch.cat(predictions, dim=0)
    return predictions


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Set random seed
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = pickle.load(open("checkpoints/dataset/imputed_dataset.pkl", "rb"))

    # Create unimputed dataset
    unimputed_dataset = pickle.load(open("checkpoints/dataset/dataset.pkl", "rb"))

    # Create train, test, val splits
    train_dataset, val_dataset, test_dataset = make_train_test_split(
        dataset, 0.8, 0.1, 0.1
    )

    # Create dataloaders
    seq_len = 10
    pred_len = 5
    param = "Mean RR"

    train_dataset = HRVDataset(train_dataset, seq_len, pred_len, param)
    val_dataset = HRVDataset(val_dataset, seq_len, pred_len, param)
    test_dataset = HRVDataset(test_dataset, seq_len, pred_len, param)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # Create model
    architecture = {
        "d_model": 128,
        "n_heads": 8,
        "d_hidden": 256,
        "n_enc_layers": 4,
        "n_dec_layers": 4,
        "dropout": 0.1,
        "n_phases": 8,
    }
    model = ContinuousTransformer(architecture, seq_len, pred_len, 150, 4, 11)
    model.to(device)

    # Collect info for loss curve plot
    train_losses = []
    val_losses = []
    train_iters = []
    val_iters = []
    min_val_loss = sys.maxsize

    # Train model
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10
    for epoch in range(num_epochs):
        for i, (inputs, targets) in enumerate(train_loader):
            # Forward pass
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            train_losses.append(loss.item())
            train_iters.append(epoch * len(train_loader) + i)

            # Compute validation loss
            val_loss = 0
            for j, (val_inputs, val_targets) in enumerate(val_loader):
                val_inputs = val_inputs.to(device)
                val_targets = val_targets.to(device)
                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_targets)
            val_loss /= len(val_loader)
            val_losses.append(val_loss.item())
            val_iters.append(epoch * len(train_loader) + i)

            # Take gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(
                    f"Epoch {epoch}, step {i}, train loss {loss.item()}, val loss {val_loss.item()}"
                )

    # Get test loss
    test_loss = 0
    for i, (test_inputs, test_targets) in enumerate(test_loader):
        test_inputs = test_inputs.to(device)
        test_targets = test_targets.to(device)
        test_outputs = model(test_inputs)
        test_loss += criterion(test_outputs, test_targets)

    print(f"Test loss: {test_loss / len(test_loader)}")

    # Save model
    if not os.path.isdir("checkpoints/transformer"):
        os.makedirs("checkpoints/transformer")
    model.save("checkpoints/transformer/transformer.pt")

    # Get predictions
    predictions = predict(model, test_loader, device)

    # Plot train and val losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_iters, train_losses, label="train loss")
    plt.plot(val_iters, val_losses, label="val loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    if not os.path.isdir("plots/transformer"):
        os.makedirs("plots/transformer")
    plt.savefig("plots/transformer/loss.png")

    # Plot predictions
    print("Plotting predictions")
    plot_predictions(
        model,
        predictions,
        test_dataset,
        unimputed_test_dataset,
        args.seq_len,
        args.pred_len,
        plots_dir="./plots/5min/predictions/linear_nn",
        param=param,
    )

    # Plot scatterplots
    print("Plotting scatterplots")
    generate_scatterplots(
        model=model,
        param_name=param,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        predictions=predictions,
        test_dataset=unimputed_test_dataset,
        plots_dir="./plots/5min/scatterplots/transformer",
    )
