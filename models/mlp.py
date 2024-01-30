import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
from utils.config import config


class MLP(nn.Module):
    """
    Simple MLP for testing purposes
    Input dim is (batch_size, num_actors + num_temporal_features, seq_len)
    Output dim is (batch_size, num_actors, pred_len) since we only predict HRV value
    """

    def __init__(self, model_config):
        super().__init__()
        self.batch_size = model_config["batch_size"]
        self.num_actors = len(config.role_names)
        self.seq_len = model_config["seq_len"]
        self.pred_len = model_config["pred_len"]
        self.input_size = self.num_actors * self.seq_len + len(config.phases) + 2
        self.hidden_dims = model_config["hidden_dims"]
        self.output_size = self.num_actors * self.pred_len
        self.model_name = model_config["model_name"]
        self.input_layer = nn.Linear(self.input_size, self.hidden_dims[0])
        for i in range(len(self.hidden_dims) - 1):
            setattr(
                self, f"fc{i}", nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1])
            )
            setattr(self, f"bn{i}", nn.BatchNorm1d(self.hidden_dims[i + 1]))
        self.output_layer = nn.Linear(self.hidden_dims[-1], self.output_size)
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            # Initialize linear layers with Kaiming initialization
            init.kaiming_uniform_(module.weight)
            if module.bias is not None:
                init.constant_(module.bias, 0)

    def forward(self, x):
        # Input dim is (batch_size, input_size, seq_len)
        # Reshape to (batch_size, seq_len * num_actors * num_features)
        # Output dim is (batch_size, num_actors, pred_len) since we only predict HRV value
        p = 0.1
        batch_size = x.shape[0]
        num_actors = config.num_actors
        hrv_in = x[:, 0:num_actors, :].reshape(batch_size, -1)
        temporal_in = x[:, num_actors:, -1].reshape(batch_size, -1)
        x = torch.cat((hrv_in, temporal_in), dim=1)
        x = x.view(batch_size, -1)
        x = self.input_layer(x)
        x = F.relu(x)
        for i in range(len(self.hidden_dims) - 1):
            x = getattr(self, f"fc{i}")(x)
            x = F.relu(x)
            x = F.dropout(x, p=p)
            x = getattr(self, f"bn{i}")(x)
        x = self.output_layer(x)
        x = x.view(batch_size, self.num_actors, self.pred_len)
        return x
