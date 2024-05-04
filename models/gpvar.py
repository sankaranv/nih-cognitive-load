from gluonts.mx import Trainer, GPVAREstimator
from utils.config import config
import numpy as np
import pickle
import os
from tqdm import tqdm


class GPVar:
    def __init__(self, model_config):
        self.base_model = model_config["base_model"]
        self.model_name = model_config["model_name"]
        self.seq_len = model_config["seq_len"]
        self.pred_len = 1
        self.num_actors = len(config.role_names)
        self.model_config = model_config
        self.trained = False
        self.num_temporal_features = config.num_phases + 1
        # Setup models
        self.models = {}
        self.setup_models()

    def setup_models(self):
        for param in config.param_names:
            self.models[param] = GPVAREstimator(
                freq="T",
                prediction_length=self.seq_len,
                target_dim=self.num_actors + self.num_temporal_features,
                context_length=None,
                trainer=Trainer(
                    ctx=self.model_config["device"],
                    epochs=self.model_config["num_epochs"],
                    learning_rate=self.model_config["lr"],
                ),
            )

    def load(self, filename):
        self.models = pickle.load(open(filename, "rb"))
        self.trained = True

    def save(self, save_path):
        if not self.trained:
            raise ValueError("Dependency network is not trained, cannot save models")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(os.path.dirname(os.path.join(save_path, "anomaly"))):
            os.makedirs(os.path.dirname(os.path.join(save_path, "anomaly")))
        # Get first key from models
        param_key = list(self.models.keys())[0]
        role_key = list(self.models[param_key].keys())[0]
        model_name = self.models[param_key][role_key].__class__.__name__
        if not os.path.exists(os.path.join(save_path, "anomaly")):
            os.makedirs(os.path.join(save_path, "anomaly"))
        pickle.dump(
            self.models,
            open(f"{save_path}/anomaly/{model_name}_dependency_network.pkl", "wb"),
        )

    def predict_proba(self, dataset, burn_in, max_iter, logging_freq=10, verbose=True):
        pass

    def train(self, dataset, verbose=True):
        input_output_pairs = self.create_input_output_pairs(dataset)
        print(
            input_output_pairs["Mean RR"][0][0].shape,
            input_output_pairs["Mean RR"][0][1].shape,
        )

    def get_individual_features(self, input_output_pairs, param, role):
        # Create training data by concatenating remaining features to the existing input vector
        X = []
        y = []
        actor_idx = config.role_colors[role]
        for sample in input_output_pairs[param]:
            target = sample[1][actor_idx][0]
            remaining_actors_data = np.delete(sample[1][:, 0].reshape(-1), actor_idx)
            input_vector = np.append(
                sample[0][:, 0, :].reshape(-1),
                remaining_actors_data.reshape(-1),
            )
            # Add temporal features
            input_vector = np.append(input_vector, sample[1][actor_idx][1:].reshape(-1))
            X.append(input_vector)
            y.append(target)
        return np.array(X), np.array(y)

    def create_input_output_pairs(self, dataset):
        num_features = self.num_actors + self.num_temporal_features
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
                        # The shape of the input vector is (num_actors, 1 + num_temporal_features, seq_len)
                        # We want a vector of shape (num_actors + num_temporal_features, seq_len)
                        # Take HRV from all timesteps and temporal features from the most recent timestep
                        input_hrv = input_vector[:, 0, :]
                        input_temporal = input_vector[:, 1:, -1]
                        input_vector = np.concatenate(
                            (input_hrv, input_temporal), axis=0
                        )
                        output_vector = param_data[:, :, i + self.seq_len]
                        assert np.all(~np.isnan(output_vector))
                        input_output_pairs[key].append((input_vector, output_vector))
        return input_output_pairs
