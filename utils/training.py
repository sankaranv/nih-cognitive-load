import torch
from utils.data import cv_split
from utils.config import config


def train(model, train_loader, val_loader, num_epochs, optimizer, criterion, device, verbose=False):
    trace = {"train_loss": [], "val_loss": []}
    for epoch in range(num_epochs):
        train_loss = train_step(model, train_loader, optimizer, criterion, device)
        trace["train_loss"].append(train_loss)
        if val_loader is not None:
            val_loss = val(model, val_loader, criterion, device)
            trace["val_loss"].append(val_loss)
        if verbose:
            print(f"Epoch {epoch + 1}: Train Loss: {train_loss}, Val Loss: {val_loss}")
    return trace

def train_step(model, train_loader, optimizer, criterion, device):

    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model.forward(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        # wandb.log({"Train Loss": loss.item()})
    return train_loss / len(train_loader.dataset)


def val(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model.forward(data)
            val_loss += criterion(output, target).item()
    val_loss /= len(val_loader.dataset)
    # wandb.log({"Val Loss": val_loss})
    return val_loss


def test(model, test_loader, optimizer, criterion, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model.forward(data)
            test_loss += criterion(output, target).item()
    test_loss /= len(test_loader.dataset)
    # wandb.log({"Test Loss": test_loss})
    return test_loss


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


def create_model(model_type, model_config):
    if model_type == "JointLinear":
        from models.joint_linear import JointLinearModel

        model = JointLinearModel(model_config)
    elif model_type == "ParameterFreeAutoregressive":
        from models.param_free import ParameterFreeAutoregressiveModel

        model = ParameterFreeAutoregressiveModel(model_config)
    elif model_type == "JointNNModel":
        from models.torch_model import JointNNModel

        model = JointNNModel(model_config)
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    return model


def cross_validation(model_type, model_config, dataset, num_folds=5, verbose=False):
    # Create train, test, val splits
    cv_splits = cv_split(dataset, num_folds=num_folds)
    cv_traces = []
    for i, (train_dataset, val_dataset, test_dataset) in enumerate(cv_splits):
        print(f"Running CV split {i+1}")
        print(f"Train cases: {train_dataset.keys()}")
        print(f"Val cases: {val_dataset.keys()}")
        print(f"Test cases: {test_dataset.keys()}")

        key = list(dataset.keys())[0]
        num_features = dataset[key].shape[-2]

        # Train model and get traces
        cv_model = create_model(model_type, model_config)
        trace = cv_model.train(train_dataset, val_dataset, verbose)
        trace = cv_model.predict(test_dataset, trace)
        cv_traces.append(trace)

    # Create combined trace that collects data from all cases
    combined_trace = {param: {} for param in config.param_names}
    for param in config.param_names:
        for key in cv_traces[0][param].keys():
            combined_trace[param][key] = {}
    # Combine traces across CV splits
    for k, trace in enumerate(cv_traces):
        for param_name, param_trace in trace.items():
            for key in trace[param_name].keys():
                # For losses, save one dict entry per CV split
                if key in ["train_loss", "val_loss", "test_loss"]:
                    combined_trace[param_name][key][k] = trace[param_name][key]
                else:
                    # For predictions and metrics, put all cases from different CV splits into one dict
                    for case_idx, case_trace in trace[param_name][key].items():
                        if case_idx not in combined_trace[param_name][key]:
                            combined_trace[param_name][key][case_idx] = case_trace
                        else:
                            print(
                                f"Cross-validation: case {case_idx} param {param_name} {key} found in multiple CV splits"
                            )

    # Train model on full dataset, we can ignore trace
    model = create_model(model_type, model_config)
    _ = model.train(dataset)

    return model, combined_trace
