import json
import os


class Config:
    def __init__(self):
        self.load_config()

    def load_config(self, config_path="config.json"):
        with open(config_path) as f:
            config = json.load(f)

        # Set global variables based on JSON
        self.param_names = config["param_names"]
        self.temporal_feature_names = config["temporal_feature_names"]
        self.static_feature_names = config["static_feature_names"]
        self.param_indices = config["param_indices"]
        self.role_names = config["role_names"]
        self.role_indices = {role: i for i, role in enumerate(self.role_names)}
        self.role_colors = config["role_colors"]
        self.num_actors = len(self.role_names)
        self.phases = config["phases"]
        self.num_phases = len(self.phases)
        self.valid_cases = config["valid_cases"]

    def update_config(self, config_path="config.json"):
        # Update attributes that are not in the config file
        self.role_indices = {role: i for i, role in enumerate(self.role_names)}
        self.num_actors = len(self.role_names)
        self.num_phases = len(self.phases)

        # Rewrite the config file with the current values
        config = {
            "param_names": self.param_names,
            "temporal_feature_names": self.temporal_feature_names,
            "static_feature_names": self.static_feature_names,
            "param_indices": self.param_indices,
            "role_names": self.role_names,
            "role_colors": self.role_colors,
            "phases": self.phases,
            "valid_cases": self.valid_cases,
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)


# Create an instance of Config to load the configuration
config = Config()
