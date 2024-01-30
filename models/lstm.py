import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
from utils.config import config

class LSTM(nn.Module):
    """
    Simple MLP for testing purposes
    Input dim is (batch_size, num_actors + num_temporal_features, seq_len)
    Output dim is (batch_size, num_actors, pred_len) since we only predict HRV value
    """

    def __init__(
        self, model_config
    ):
        super().__init__()
        self.batch_size = model_config["batch_size"]
        self.num_actors = len(config.role_names)
        self.seq_len = model_config["seq_len"]
        self.pred_len = model_config["pred_len"]
        self.input_size = self.num_actors + len(config.phases) + 2
        self.embedding_dim = model_config["embedding_dim"]
        self.hidden_dim = model_config["hidden_dim"]
        self.output_size = self.num_actors * self.pred_len
        self.model_name = model_config["model_name"]

        # Create layers of LSTM model
        self.lstm = nn.LSTM(self.input_size, self.hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(self.hidden_dim, self.output_size)
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            # Initialize linear layers with Kaiming initialization
            init.kaiming_uniform_(module.weight)
            if module.bias is not None:
                init.constant_(module.bias, 0)

    def forward(self, x):
        # Input dim is (batch_size, input_size, seq_len)
        # Reshape to (batch_size, seq_len, input_size)
        x = x.permute(0, 2, 1)
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        _, (hn, _) = self.lstm(x)
        out = self.output_layer(hn)
        out = out.view(batch_size, self.num_actors, self.pred_len)
        return out


