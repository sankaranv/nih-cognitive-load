import torch

from models.joint_linear import JointLinearRegressor, JointLinearClassifier
from models.torch_model import JointNNRegressor, JointNNClassifier
from models.param_free import (
    ParameterFreeAutoregressiveModel,
    ParameterFreeAutoregressiveClassifier,
    RandomRegressionModel,
)
from models.gaussian_process import GaussianProcessAnomalyDetector
from models.dependency_network import (
    DependencyNetwork,
    IndependentComponentDependencyNetwork,
)
from models.conditional_vae import ConditionalVAEAnomalyDetector
from utils.data import *
from utils.config import config
from visualization.all_plots import *
from sklearn.metrics import roc_curve, auc


def train(
    model,
    train_loader,
    val_loader,
    num_epochs,
    optimizer,
    criterion,
    device,
    verbose=False,
):
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
    if model_type == "JointLinearRegressor":
        model = JointLinearRegressor(model_config)
    elif model_type == "JointLinearClassifier":
        model = JointLinearClassifier(model_config)
    elif model_type == "ParameterFreeAutoregressiveModel":
        model = ParameterFreeAutoregressiveModel(model_config)
    elif model_type == "ParameterFreeAutoregressiveClassifier":
        model = ParameterFreeAutoregressiveClassifier(model_config)
    elif model_type == "JointNNRegressor":
        model = JointNNRegressor(model_config)
    elif model_type == "JointNNClassifier":
        model = JointNNClassifier(model_config)
    elif model_type == "RandomRegressor":
        model = RandomRegressionModel(model_config)
    elif model_type == "DependencyNetwork":
        model = DependencyNetwork(model_config)
    elif model_type == "IndependentComponentDependencyNetwork":
        model = IndependentComponentDependencyNetwork(model_config)
    elif model_type == "ConditionalVAEAnomalyDetector":
        model = ConditionalVAEAnomalyDetector(model_config)
    elif "GaussianProcess" in model_type:
        model = GaussianProcessAnomalyDetector(model_config)
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
    model_name = model_config["model_name"]
    print(f"Training {model_name} model on full dataset")
    model = create_model(model_type, model_config)
    _ = model.train(dataset, verbose=False)

    return model, combined_trace


def anomaly_detection_eval(
    model_config,
    full_dataset,
    num_anomalies=5,
    seq_len=3,
    num_folds=5,
    verbose=False,
):
    model_dir = "./checkpoints"
    plots_dir = "./plots"

    # Create train, test, val splits
    cv_splits = cv_split(full_dataset, num_folds=num_folds, val=False)
    cv_traces = []
    for i, (train_dataset, val_dataset, test_dataset) in enumerate(cv_splits):
        print(f"Running CV split {i+1}")
        print(f"Train cases: {train_dataset.keys()}")
        print(f"Val cases: {val_dataset.keys()}")
        print(f"Test cases: {test_dataset.keys()}")
        model_name = model_config["model_name"]

        # Train model and get probabilities
        model = create_model(model_config["model_name"], model_config)

        model.train(train_dataset, verbose=verbose)
        model.save(model_dir)

        # For each test case, sample num_anomalies points from the other test cases and swap them into random positions
        predictions = {}
        num_params = len(config.param_names)
        test_cases = list(test_dataset.keys())
        for case_idx in test_dataset.keys():
            num_samples = test_dataset[case_idx].shape[-1]
            test_sample_positions = np.arange(seq_len, num_samples)
            predictions[case_idx] = np.ones((num_params, num_samples - seq_len))
            for param_idx in range(num_params):
                for _ in range(num_anomalies):
                    # Obtain an anomaly sample
                    anomaly_case = np.random.choice(
                        [x for x in test_cases if x != case_idx]
                    )
                    anomaly_sample_positions = np.arange(
                        seq_len, test_dataset[anomaly_case].shape[-1]
                    )
                    sample_idx = np.random.choice(anomaly_sample_positions)
                    sample_val = test_dataset[anomaly_case][param_idx, :, 0, sample_idx]

                    # Replace a random sample in the test case with the anomaly sample
                    anomaly_pos = np.random.choice(test_sample_positions)
                    test_dataset[case_idx][param_idx, :, 0, anomaly_pos] = sample_val
                    predictions[case_idx][param_idx, anomaly_pos - seq_len] = 0

        # Predict probabilities on the modified test dataset
        print(f"Obtaining test set probabilities for CV split {i+1}")
        test_probs, trace = model.predict_proba(test_dataset)

        # for j in test_dataset.keys():
        #     check = np.array_equal(
        #         test_dataset[j][:, :, 0, :], np.round(test_dataset[j][:, :, 0, :])
        #     )
        #     print(f"Case {j} is integer: {check}")

        if not os.path.exists(f"{model_dir}/anomaly/eval/{model_name}"):
            os.makedirs(f"{model_dir}/anomaly/eval/{model_name}")
        with open(f"{model_dir}/anomaly/eval/{model_name}/cv{i}_probs.pkl", "wb") as f:
            pickle.dump(test_probs, f)

        # Plot ROC curves for anomalies
        print(f"Plotting ROC curves for CV split {i+1} aggregated")
        plot_anomaly_roc(
            test_probs, predictions, model_name, plots_dir, per_case=False, cv_idx=i
        )

        print(f"Plotting ROC curves for CV split {i + 1} per case")
        plot_anomaly_roc(test_probs, predictions, model_name, plots_dir, per_case=True)

        # Plot probabilities for each test case
        print(f"Plotting probabilities for CV split {i+1}")
        plot_anomalies(
            test_dataset,
            test_probs,
            model_name,
            seq_len,
            plots_dir,
        )
