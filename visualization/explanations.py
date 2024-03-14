import numpy as np
import shap
import torch

from models.joint_linear import JointLinearModel
from utils.create_batches import create_torch_loader_from_dataset
from utils.config import config
from matplotlib import pyplot as plt
import os


def shap_explain(model: JointLinearModel, dataset, seq_len, plots_dir, num_samples=100):
    for param in config.param_names:
        if model.__class__.__name__ == "JointLinearRegressor":
            # Prepare dataset
            X, y = model.create_train_dataset(dataset, param)

            # Subsample for speed using shap.sample
            X = shap.sample(X, num_samples)

            # Create explainer
            explainer = shap.KernelExplainer(model.models[param].predict, X)
            shap_values = explainer.shap_values(X)

            # Take out only surgeon feature
            shap_values = shap_values[-1]

        elif model.__class__.__name__ == "JointNNRegressor":
            # Prepare dataset
            data_loader = create_torch_loader_from_dataset(
                dataset,
                model.seq_len,
                model.pred_len,
                param,
                model.model_config["batch_size"],
                shuffle=True,
            )

            # Sample from the data loader
            if model.model_config["batch_size"] > num_samples:
                X, y = next(iter(data_loader))
                X = X[:num_samples]
            else:
                X, y = next(iter(data_loader))
                while len(X) < num_samples:
                    next_X, next_y = next(iter(data_loader))
                    X = torch.cat((X, next_X))
                    y = torch.cat((y, next_y))
                X = X[:num_samples]

            # Create explainer
            explainer = shap.DeepExplainer(model.models[param], X)
            shap_values = explainer.shap_values(X, check_additivity=False)
            shap_values = np.array(shap_values[-1])

            # Create feature vector
            # Shap values and X are of shape (num_samples, num_actors + num_phases + 1, seq_len)
            # We want vectors of shape (num_samples, num_actors * seq_len + num_phases + 1)
            shap_temporal_features = shap_values[:, config.num_actors :, -1]
            shap_hrv_features = shap_values[:, : config.num_actors, :].reshape(
                num_samples, -1
            )
            shap_values = np.concatenate(
                (shap_hrv_features, shap_temporal_features), axis=1
            )

            X_temporal_features = X[:, config.num_actors :, -1]
            X_hrv_features = X[:, : config.num_actors, :].reshape(num_samples, -1)
            X = np.concatenate((X_hrv_features, X_temporal_features), axis=1)

        # Create feature names
        feature_names = []
        for i in range(seq_len, 0, -1):
            for actor_name in config.role_names:
                feature_names.append(f"{actor_name} t-{i}")
        for p in config.phases:
            feature_names.append(f"Phase {p}")
        feature_names.append("No Phase")
        feature_names.append("Time")

        # Plot
        shap.summary_plot(shap_values, X, feature_names=feature_names, max_display=27)
