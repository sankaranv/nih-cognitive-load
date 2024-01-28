import torch
from utils.data import cv_split
from utils.config import config


def train(model, train_loader, optimizer, criterion, device):
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
        val(model, train_loader, optimizer, criterion, device)
    return train_loss / len(train_loader.dataset)


def val(model, train_loader, optimizer, criterion, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model.forward(data)
            val_loss += criterion(output, target).item()
    val_loss /= len(train_loader.dataset)
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
        cv_model.train(train_dataset, verbose)
        trace = cv_model.predict(test_dataset)
        cv_traces.append(trace)

    # Combine traces across CV splits
    combined_trace = {}
    for trace in cv_traces:
        for case_idx, case_trace in trace.items():
            if case_idx not in combined_trace:
                combined_trace[case_idx] = {}
            for param_name, param_trace in case_trace.items():
                if param_name not in combined_trace[case_idx]:
                    combined_trace[case_idx][param_name] = param_trace
                else:
                    print(
                        f"Cross-validation: case {case_idx} param {param_name} found in multiple CV splits"
                    )

    # Train model on full dataset
    model = create_model(model_type, model_config)
    model.train(dataset)

    return model, combined_trace
