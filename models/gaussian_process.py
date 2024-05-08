from utils.config import config
import numpy as np
import os
import GPy
from tqdm import tqdm


class GaussianProcessAnomalyDetector:
    def __init__(self, model_config):
        super().__init__()
        self.num_actors = len(config.role_names)
        self.seq_len = model_config["seq_len"]
        self.pred_len = model_config["pred_len"]
        self.model_name = model_config["model_name"]
        self.input_size = self.num_actors * self.seq_len + len(config.phases) + 2
        self.output_size = self.num_actors * self.pred_len
        self.trained = False
        self.independent = model_config["independent"]
        self.setup_models(kernel=model_config["kernel"])

    def setup_models(self, kernel="rbf"):
        # Create a kernel for each HRV parameter
        if kernel == "rbf":
            self.kernels = {
                param: GPy.kern.RBF(input_dim=self.input_size, ARD=True)
                for param in config.param_names
            }

        self.models = {
            param: GPy.models.GPRegression(
                np.zeros((1, self.input_size)),
                np.zeros((1, self.output_size)),
                kernel=self.kernels[param],
                normalizer=True,
            )
            for param in config.param_names
        }

    def save(self, path):
        pass
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # for param in config.param_names:
        #     for actor in config.role_names:
        #         self.models[param][actor].save_model(os.path.join(path, f"{param}.gpy"))

    def create_dataset(self, input_output_pairs, param):
        # Create training data by concatenating remaining features to the existing input vector
        X = []
        y = []
        for sample in input_output_pairs[param]:
            target = sample[1][:, 0].reshape(-1)
            input_vector = sample[0][:, 0, :].reshape(-1)
            # Add temporal features for current timestep
            input_vector = np.append(input_vector, sample[1][0, 1:].reshape(-1))
            X.append(input_vector)
            y.append(target)
        return np.array(X), np.array(y)

    def create_input_output_pairs(self, dataset):
        input_output_pairs = {}
        for key in config.param_names:
            input_output_pairs[key] = []
        for case_idx, case_data in dataset.items():
            # Shape of the data is (num_params, num_actors, num_features, num_timesteps)
            for param_idx, param_data in enumerate(case_data):
                # Shape of the data is (num_actors, num_features, num_timesteps)
                # Timestep axis is moved to the front for broadcasting
                num_timesteps = param_data.shape[-1]
                param_data = np.transpose(param_data, (2, 0, 1))
                # Check if there are any NaNs in any feature for any of the actors
                # Shape of the mask is (num_timesteps,)
                mask = np.all(~np.isnan(param_data), axis=(1, 2))
                param_data = np.transpose(param_data, (1, 2, 0))
                # Use the mask to check if there are any missing values in the last L + 1 timesteps
                for i in range(num_timesteps - self.seq_len - 1):
                    if np.all(mask[i : i + self.seq_len + 1]):
                        # If there are no missing values, add to training set
                        # The first L steps are used as input and the last one is used as output
                        key = config.param_names[param_idx]
                        input_vector = param_data[:, :, i : i + self.seq_len]
                        output_vector = param_data[:, :, i + self.seq_len]
                        assert np.all(~np.isnan(output_vector))
                        input_output_pairs[key].append((input_vector, output_vector))
        return input_output_pairs

    def train(self, dataset, verbose=False):
        input_output_pairs = self.create_input_output_pairs(dataset)
        for param in config.param_names:
            print(f"Training GP model for {param}")
            X, y = self.create_dataset(input_output_pairs, param)
            self.models[param].set_XY(X, y)
            self.models[param].optimize(messages=verbose)
        self.trained = True

    def predict_proba(self, dataset, verbose=False):
        if not self.trained:
            raise ValueError("Models are not trained, cannot do inference")
        # Object for collecting samples
        probs = {}
        for case_idx, case_data in dataset.items():
            probs[case_idx] = {}
            num_samples = case_data.shape[-1]
            for param in config.param_names:
                probs[case_idx][param] = np.zeros(
                    (config.num_actors, num_samples - self.seq_len)
                )

        # Predict probabilities for each test point after first seq_len steps
        trace = {param: {"mean": {}, "var": {}} for param in config.param_names}
        for case_idx, case_data in tqdm(dataset.items()):
            case_input_output_pairs = self.create_input_output_pairs(
                {case_idx: case_data}
            )
            for param in config.param_names:
                X, y = self.create_dataset(case_input_output_pairs, param)
                trace[param]["mean"][case_idx] = np.zeros(
                    (X.shape[0], config.num_actors)
                )
                trace[param]["var"][case_idx] = np.zeros(
                    (X.shape[0], config.num_actors)
                )
                for i in range(X.shape[0]):
                    input_vector = X[i]
                    target = y[i]
                    # Predict the posterior mean and variance
                    means, variances = self.models[param].predict(
                        input_vector.reshape(1, -1)
                    )

                    log_probs = self.models[param].log_predictive_density(
                        target.reshape(1, -1),
                        input_vector.reshape(1, -1),
                    )
                    print(log_probs.shape)
                    probs[case_idx][param][:, i] = log_probs

        if verbose:
            print("Predicted probabilities for HRV dataset")

        return probs, trace
