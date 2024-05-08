import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import BayesianRidge
from utils.config import config
import pickle
import os
from tqdm import tqdm
from sklearn.model_selection import cross_val_score


class DependencyNetwork:
    def __init__(self, model_config):
        self.base_model = model_config["base_model"]
        self.model_name = model_config["model_name"]
        self.seq_len = model_config["seq_len"]
        self.model_config = model_config
        if not model_config["independent"]:
            self.burn_in = model_config["burn_in"]
            self.max_iter = model_config["max_iter"]
        self.trained = False
        # Setup models
        self.models = {}
        self.setup_models()

    def setup_models(self):
        for param in config.param_names:
            self.models[param] = {}
            for actor in config.role_names:
                if self.base_model == "BayesianRidge":
                    self.models[param][actor] = BayesianRidge()
                elif self.base_model == "RandomForestClassifier":
                    self.models[param][actor] = RandomForestClassifier()

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

    def predict_proba(self, dataset, logging_freq=100, verbose=False):
        if not self.trained:
            raise ValueError("Models are not trained, cannot do inference")
        print(f"Begin MCMC for {self.max_iter} iterations and {self.burn_in} burn-in")
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
        trace = {}
        for case_idx, case_data in tqdm(dataset.items()):
            for param_idx, param_data in enumerate(case_data):
                param = config.param_names[param_idx]

                # Shape of the data is (num_actors, num_features, num_timesteps)
                for i in range(param_data.shape[-1] - self.seq_len):
                    # Take L timesteps as input vector
                    input_vector = param_data[:, :, i : i + self.seq_len]

                    # Take the next timestep as output vector
                    output_vector = param_data[:, :, i + self.seq_len]

                    input_vector = np.copy(input_vector)
                    output_vector = np.copy(output_vector)

                    prev_output_vector = np.zeros(output_vector.shape)
                    # Gibbs sampling for predicting probabilities
                    for t in range(self.max_iter + self.burn_in):
                        # Sample each actor's HRV value from its conditional distribution
                        for actor in config.role_names:
                            actor_idx = config.role_colors[actor]
                            # Add HRV features from remaining actors
                            remaining_actors_data = np.delete(
                                output_vector[:, 0].reshape(-1), actor_idx, axis=0
                            )
                            input_vector_for_actor = np.append(
                                input_vector[:, 0, :].reshape(-1),
                                remaining_actors_data.reshape(-1),
                            )
                            # Add temporal features
                            input_vector_for_actor = np.append(
                                input_vector_for_actor,
                                output_vector[actor_idx][1:].reshape(-1),
                            )
                            # Predict value using regressor
                            input_vector_for_actor = input_vector_for_actor.reshape(
                                1, -1
                            )
                            actor_pred = self.models[param][actor].predict(
                                input_vector_for_actor
                            )[0]

                            # Get the posterior probability distribution of the predicted value
                            actor_prob_dist = self.models[param][actor].predict_proba(
                                input_vector_for_actor
                            )

                            # Obtain the probability of the ground truth HRV instead of predicted HRV
                            actual_hrv = int(output_vector[actor_idx, 0])
                            if actual_hrv >= actor_prob_dist.shape[1]:
                                actual_hrv = actor_prob_dist.shape[1] - 1
                            # print(
                            #     f"Predicted HRV: {actor_pred}, Actual HRV: {actual_hrv}, Shape: {actor_prob_dist.shape}"
                            # )
                            actor_prob = actor_prob_dist[0][actual_hrv]
                            # print(
                            #     f"Predicted HRV: {actor_pred}, Actual HRV: {actual_hrv}, Probability: {actor_prob}"
                            # )

                            prev_output_vector[actor_idx] = output_vector[actor_idx]
                            output_vector[actor_idx] = actor_prob
                            # Store samples after burn-in
                            if t >= self.burn_in:
                                probs[case_idx][param][actor_idx, i] = actor_prob

                        # Log progress
                        if t % logging_freq == 0 and verbose:
                            if t < self.burn_in:
                                print(
                                    f"Case {case_idx} {param} Timestep {i} Burn-in {t}"
                                )
                            else:
                                print(
                                    f"Case {case_idx} {param} Timestep {i} Iteration {t - self.burn_in}"
                                )

                        # Check if values have converged and stop MCMC if so
                        if (
                            np.allclose(prev_output_vector, output_vector)
                            and t > self.burn_in
                        ):
                            # # Pad the rest of the samples with the last value
                            # for j in range(t - burn_in, max_iter):
                            #     for actor_idx in range(output_vector.shape[0]):
                            #         probs[case_idx][param][
                            #             actor_idx, i, j
                            #         ] = output_vector[actor_idx, 0]
                            break

                # Check if integer
                check = np.array_equal(
                    param_data[:, 0, :], np.round(param_data[:, 0, :])
                )
                print(f"Case {case_idx} {param} is integer: {check}")

        if verbose:
            print("Predicted probabilities for HRV dataset")

        return probs, trace

    def train(self, dataset, verbose=True, cv=False):
        # Create input-output pairs for each parameter
        input_output_pairs = self.create_input_output_pairs(dataset)
        # Train models for each parameter
        for param in config.param_names:
            for role in config.role_names:
                X, y = self.get_individual_features(input_output_pairs, param, role)
                self.models[param][role].fit(X, y)
                # Cross validation
                if cv:
                    score = cross_val_score(
                        self.models[param][role],
                        X,
                        y,
                        cv=5,
                        scoring="neg_mean_squared_error",
                    )
                    if verbose:
                        print(
                            f"Cross-validation MSE for {param} {role} model => Mean: {-score.mean():.4f} Max: {-score.min():.4f} Min: {-score.max():.4f}"
                        )
        self.trained = True

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


class IndependentComponentDependencyNetwork(DependencyNetwork):
    def __init__(self, model_config):
        super().__init__(model_config)

    def get_individual_features(self, input_output_pairs, param, role):
        # Create training data by concatenating remaining features to the existing input vector
        X = []
        y = []
        actor_idx = config.role_colors[role]
        for sample in input_output_pairs[param]:
            target = sample[1][actor_idx][0]
            input_vector = sample[0][:, 0, :].reshape(-1)
            # Add temporal features
            input_vector = np.append(input_vector, sample[1][actor_idx][1:].reshape(-1))
            X.append(input_vector)
            y.append(target)
        return np.array(X), np.array(y)

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
        trace = {}
        for case_idx, case_data in tqdm(dataset.items()):
            for param_idx, param_data in enumerate(case_data):
                param = config.param_names[param_idx]
                # Shape of the data is (num_actors, num_features, num_timesteps)
                for i in range(param_data.shape[-1] - self.seq_len):
                    # Take L timesteps as input vector
                    input_vector = param_data[:, :, i : i + self.seq_len]
                    input_vector = np.copy(input_vector)
                    # Take the next timestep as output vector
                    output_vector = param_data[:, :, i + self.seq_len]
                    output_vector = np.copy(output_vector)
                    prev_output_vector = np.zeros(output_vector.shape)

                    # Sample each actor's HRV value from its conditional distribution
                    for actor in config.role_names:
                        actor_idx = config.role_colors[actor]
                        input_vector_for_actor = input_vector[:, 0, :].reshape(-1)
                        # Add temporal features
                        input_vector_for_actor = np.append(
                            input_vector_for_actor,
                            output_vector[actor_idx][1:].reshape(-1),
                        )
                        # Predict value using regressor
                        input_vector_for_actor = input_vector_for_actor.reshape(1, -1)
                        actor_pred = self.models[param][actor].predict(
                            input_vector_for_actor
                        )[0]

                        # Get the posterior probability distribution of the predicted value
                        actor_prob_dist = self.models[param][actor].predict_proba(
                            input_vector_for_actor
                        )

                        # Obtain the probability of the ground truth HRV instead of predicted HRV
                        actual_hrv = int(output_vector[actor_idx, 0])
                        if actual_hrv >= actor_prob_dist.shape[1]:
                            actual_hrv = actor_prob_dist.shape[1] - 1
                        # print(
                        #     f"Predicted HRV: {actor_pred}, Actual HRV: {actual_hrv}, Shape: {actor_prob_dist.shape}"
                        # )
                        actor_prob = actor_prob_dist[0][actual_hrv]
                        # print(
                        #     f"Predicted HRV: {actor_pred}, Actual HRV: {actual_hrv}, Probability: {actor_prob}"
                        # )

                        prev_output_vector[actor_idx] = output_vector[actor_idx]
                        output_vector[actor_idx] = actor_prob
                        probs[case_idx][param][actor_idx, i] = actor_prob

        if verbose:
            print("Predicted probabilities for HRV dataset")

        return probs, trace
