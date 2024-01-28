import numpy as np
from sklearn.linear_model import BayesianRidge

from sklearn.model_selection import cross_val_score
import pickle
from utils.data import *
import argparse
from matplotlib import pyplot as plt
from matplotlib import ticker
from utils.config import config


class MCMCImputer:
    def __init__(
        self,
        lag_length,
        base_model=BayesianRidge,
    ):
        # Data preparation hyperparameters
        self.lag_length = lag_length
        # Base model for imputation
        if base_model == "BayesianRidge":
            self.base_model = BayesianRidge
        else:
            raise ValueError(
                f"Base model {base_model} not supported, please choose from BayesianRidge"
            )
        self.trained = False

        # Store names of HRV parameters and roles for indexing purposes
        if config.param_names is None:
            self.param_names = ["PNS index", "SNS index", "Mean RR", "RMSSD", "LF-HF"]
        else:
            self.param_names = config.param_names

        if config.role_names is None:
            self.role_names = ["Anes", "Nurs", "Perf", "Surg"]
        else:
            self.role_names = config.role_names

        self.param_indices = {param: i for i, param in enumerate(self.param_names)}
        self.role_indices = {role: i for i, role in enumerate(self.role_names)}

        # Store imputation models
        self.imputation_models = {}
        for key in self.param_names:
            self.imputation_models[key] = {}
            for role in self.role_names:
                self.imputation_models[key][role] = self.base_model()

    def load(self, filename):
        self.imputation_models = pickle.load(open(filename, "rb"))
        self.trained = True

    def save(self, save_path):
        if not self.trained:
            raise ValueError("Imputation models are not trained, cannot save models")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(os.path.dirname(os.path.join(save_path, "imputation"))):
            os.makedirs(os.path.dirname(os.path.join(save_path, "imputation")))
        # Get first key from imputation models
        param_key = list(self.imputation_models.keys())[0]
        role_key = list(self.imputation_models[param_key].keys())[0]
        model_name = self.imputation_models[param_key][role_key].__class__.__name__
        pickle.dump(
            self.imputation_models,
            open(f"{save_path}/imputation/{model_name}_Imputer.pkl", "wb"),
        )

    def impute(self, dataset, burn_in, max_iter, logging_freq=10, verbose=True):
        if not self.trained:
            raise ValueError("Imputation models are not trained, cannot impute data")
        print(f"Begin imputation for {max_iter} iterations and {burn_in} burn-in")
        imputed_dataset = {}
        means = self.get_means(dataset)  # Shape is (num_params, num_actors)
        variances = self.get_stddevs(dataset)  # Shape is (num_params, num_actors)
        # Object for collecting samples
        samples = {}
        for case_idx, case_data in dataset.items():
            samples[case_idx] = {}
            for param_idx, param_data in enumerate(case_data):
                seq_len = param_data.shape[-1]
                param = self.param_names[param_idx]
                # Shape is (num_cases, num_actors, num_timesteps)
                samples[case_idx][param] = np.zeros(
                    (
                        len(self.role_names),
                        seq_len - self.lag_length,
                        max_iter,
                    )
                )

        # Impute missing values in the dataset
        for case_idx, case_data in tqdm(dataset.items()):
            imputed_dataset[case_idx] = np.copy(case_data)
            for param_idx, param_data in enumerate(imputed_dataset[case_idx]):
                # Shape of the data is (num_actors, num_features, num_timesteps)
                for i in range(param_data.shape[-1] - self.lag_length):
                    # Take L timesteps as input vector
                    input_vector = param_data[:, :, i : i + self.lag_length]
                    # For the first L timesteps, use the dataset mean and variance to impute
                    # TODO - Find a way to avoid using means!
                    if i < self.lag_length:
                        for t in range(self.lag_length):
                            mean_impute_idx = np.where(np.isnan(input_vector[:, 0, t]))
                            if len(mean_impute_idx[0] > 1):
                                input_vector[mean_impute_idx, 0, t] = means[
                                    param_idx, mean_impute_idx
                                ]
                    # Take the next timestep as output vector
                    output_vector = param_data[:, :, i + self.lag_length]
                    prev_output_vector = np.zeros(output_vector.shape)
                    # Note indices where imputation is necessary
                    out_impute_idx = np.where(np.isnan(output_vector[:, 0]))
                    if len(out_impute_idx[0] > 1):
                        # Initialize with the mean, then Gibbs sample with imputation models
                        output_vector[out_impute_idx, 0] = means[
                            param_idx, out_impute_idx
                        ]
                        # Gibbs sampling for predicting missing values
                        for t in range(max_iter + burn_in):
                            # Sample each actor's HRV value from its conditional distribution
                            for actor_idx in out_impute_idx[0]:
                                imputed_actor = self.role_names[actor_idx]
                                imputed_param = self.param_names[param_idx]
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
                                # Predict missing value
                                input_vector_for_actor = input_vector_for_actor.reshape(
                                    1, -1
                                )
                                imputed_value = self.imputation_models[imputed_param][
                                    imputed_actor
                                ].predict(input_vector_for_actor)

                                prev_output_vector[actor_idx] = output_vector[actor_idx]
                                output_vector[actor_idx] = imputed_value
                                # Store samples after burn-in
                                if t > burn_in:
                                    samples[case_idx][param][
                                        actor_idx, i, t - burn_in
                                    ] = imputed_value[0]

                            # Log progress
                            if t % logging_freq == 0 and verbose:
                                if t < burn_in:
                                    print(f"Case {case_idx} Timestep {i} Burn-in {t}")
                                else:
                                    print(
                                        f"Case {case_idx} Timestep {i} Iteration {t - burn_in}"
                                    )
                            # Check if values have converged and stop MCMC if so
                            if (
                                np.allclose(prev_output_vector, output_vector)
                                and t > burn_in
                            ):
                                # Pad the rest of the samples with the last value
                                for j in range(t - burn_in, max_iter):
                                    for actor_idx in range(output_vector.shape[0]):
                                        samples[case_idx][imputed_param][
                                            actor_idx, i, j
                                        ] = output_vector[actor_idx, 0]
                                break
                    # Assign imputed values to the dataset
                    imputed_dataset[case_idx][param_idx][
                        :, :, i + self.lag_length
                    ] = output_vector
            # For good measure, make sure there are no NaNs left in the dataset
            assert np.all(~np.isnan(imputed_dataset[case_idx]))
        if verbose:
            print("Imputation applied to HRV dataset")

        return imputed_dataset, samples

    def impute_from_first_observed(
        self, dataset, burn_in, max_iter, logging_freq=10, verbose=True
    ):
        if not self.trained:
            raise ValueError("Imputation models are not trained, cannot impute data")
        print(f"Begin imputation for {max_iter} iterations and {burn_in} burn-in")
        imputed_dataset = {}
        means = self.get_means(dataset)  # Shape is (num_params, num_actors)
        # Object for collecting samples
        samples = {}
        for case_idx, case_data in dataset.items():
            if case_idx in config.valid_cases:
                continue
            samples[case_idx] = {}
            for param_idx, param_data in enumerate(case_data):
                seq_len = param_data.shape[-1]
                param = self.param_names[param_idx]
                # Shape is (num_actors, num_timesteps, num_samples)
                samples[case_idx][param] = np.zeros(
                    (
                        len(self.role_names),
                        seq_len - self.lag_length,
                        max_iter,
                    )
                )

        # Impute missing values in the dataset
        for case_idx, case_data in tqdm(dataset.items()):
            imputed_dataset[case_idx] = np.copy(case_data)
            for param_idx, param_data in enumerate(imputed_dataset[case_idx]):
                param = self.param_names[param_idx]
                # Find the first set of L timesteps with no missing values and use that as the input vector
                # We will skip all timesteps before
                input_vector = None
                start_idx = None
                for j in range(param_data.shape[-1] - self.lag_length):
                    input_vector = param_data[:, :, j : j + self.lag_length]
                    if np.all(~np.isnan(input_vector[:, 0, :])):
                        start_idx = j
                        break
                # If there is no contiguous set of L timesteps with no missing values, skip this case
                if start_idx is None:
                    print(
                        f"No contiguous set of {self.lag_length} timesteps found in Case {case_idx} {param}"
                    )
                else:
                    # Shape of the data is (num_actors, num_features, num_timesteps)
                    for i in range(start_idx, param_data.shape[-1] - self.lag_length):
                        # Take L timesteps as input vector
                        input_vector = param_data[:, :, i : i + self.lag_length]
                        # Take the next timestep as output vector
                        output_vector = param_data[:, :, i + self.lag_length]
                        prev_output_vector = np.zeros(output_vector.shape)
                        # Note indices where imputation is necessary
                        out_impute_idx = np.where(np.isnan(output_vector[:, 0]))
                        if len(out_impute_idx[0] > 1):
                            # Initialize with the mean, then Gibbs sample with imputation models
                            output_vector[out_impute_idx, 0] = means[
                                param_idx, out_impute_idx
                            ]
                            # Gibbs sampling for predicting missing values
                            for t in range(max_iter + burn_in):
                                # Sample each actor's HRV value from its conditional distribution
                                for actor_idx in out_impute_idx[0]:
                                    imputed_actor = self.role_names[actor_idx]
                                    imputed_param = self.param_names[param_idx]
                                    # Add HRV features from remaining actors
                                    remaining_actors_data = np.delete(
                                        output_vector[:, 0].reshape(-1),
                                        actor_idx,
                                        axis=0,
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
                                    # Predict missing value
                                    input_vector_for_actor = (
                                        input_vector_for_actor.reshape(1, -1)
                                    )
                                    imputed_value = self.imputation_models[
                                        imputed_param
                                    ][imputed_actor].predict(input_vector_for_actor)

                                    prev_output_vector[actor_idx] = output_vector[
                                        actor_idx
                                    ]
                                    output_vector[actor_idx] = imputed_value
                                    # Store samples after burn-in
                                    if t > burn_in:
                                        samples[case_idx][param][
                                            actor_idx, i, t - burn_in
                                        ] = imputed_value[0]

                                # Log progress
                                if t % logging_freq == 0 and verbose:
                                    if t < burn_in:
                                        print(
                                            f"Case {case_idx} Timestep {i} Burn-in {t}"
                                        )
                                    else:
                                        print(
                                            f"Case {case_idx} Timestep {i} Iteration {t - burn_in}"
                                        )
                                # Check if values have converged and stop MCMC if so
                                if (
                                    np.allclose(prev_output_vector, output_vector)
                                    and t > burn_in
                                ):
                                    # Pad the rest of the samples with the last value
                                    for j in range(t - burn_in, max_iter):
                                        for actor_idx in range(output_vector.shape[0]):
                                            samples[case_idx][imputed_param][
                                                actor_idx, i, j
                                            ] = output_vector[actor_idx, 0]
                                    break
                        # Assign imputed values to the dataset
                        imputed_dataset[case_idx][param_idx][
                            :, :, i + self.lag_length
                        ] = output_vector
                    # For good measure, make sure there are no NaNs left in the dataset after the starting index
                    assert np.all(
                        ~np.isnan(imputed_dataset[case_idx][:, :, start_idx:])
                    )
        if verbose:
            print("Imputation applied to HRV dataset")

        return imputed_dataset, samples

    def get_means(self, dataset):
        means = np.zeros((5, config.num_actors))
        num_samples = np.zeros((5, config.num_actors))
        for _, case_data in dataset.items():
            if len(case_data.shape) == 3:
                data = case_data
            elif case_data.shape[-2] > 1:
                # If temporal or static features are present, ignore them
                data = case_data[:, :, 0, :]
            num_samples += np.sum(~np.isnan(data), axis=-1)
            means += np.sum(np.nan_to_num(data), axis=-1)
        return means / num_samples

    def get_stddevs(self, dataset):
        samples = {}
        std_devs = np.zeros((5, config.num_actors))
        for _, data in dataset.items():
            for x in range(5):
                for y in range(config.num_actors):
                    if (x, y) not in samples:
                        samples[(x, y)] = np.array([])
                    samples[(x, y)] = np.concatenate(
                        (samples[(x, y)], data[x][y][~np.isnan(data[x][y])])
                    )
        for x in range(5):
            for y in range(config.num_actors):
                std_devs[x, y] = np.std(samples[(x, y)])
        return std_devs

    def train(self, dataset, verbose=True):
        # Create input-output pairs for each parameter
        input_output_pairs = self.create_input_output_pairs(dataset)
        # Train imputation models for each parameter
        for param in self.param_names:
            for role in self.role_names:
                X, y = self.get_individual_features(input_output_pairs, param, role)
                self.imputation_models[param][role].fit(X, y)
                # Cross validation
                score = cross_val_score(
                    self.imputation_models[param][role],
                    X,
                    y,
                    cv=5,
                    scoring="neg_mean_squared_error",
                )
                if verbose:
                    print(
                        f"Cross-validation MSE for imputing {param} {role} => Mean: {-score.mean():.4f} Max: {-score.min():.4f} Min: {-score.max():.4f}"
                    )
        self.trained = True

    def get_individual_features(self, input_output_pairs, param, role):
        # Create training data by concatenating remaining features to the existing input vector
        X = []
        y = []
        actor_idx = self.role_indices[role]
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
        for key in self.param_names:
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
                for i in range(num_timesteps - self.lag_length - 1):
                    if np.all(mask[i : i + self.lag_length + 1]):
                        # If there are no missing values, add to training set
                        # The first L steps are used as input and the last one is used as output
                        key = self.param_names[param_idx]
                        input_vector = param_data[:, :, i : i + self.lag_length]
                        output_vector = param_data[:, :, i + self.lag_length]
                        assert np.all(~np.isnan(output_vector))
                        input_output_pairs[key].append((input_vector, output_vector))
        return input_output_pairs
