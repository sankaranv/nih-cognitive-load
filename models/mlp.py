import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init


class MLP(nn.Module):
    """
    Simple MLP for testing purposes
    Input dim is (batch_size, num_actors + num_temporal_features, seq_len)
    Output dim is (batch_size, num_actors, pred_len) since we only predict HRV value
    """

    def __init__(
        self, hidden_dims, batch_size, num_actors, seq_len, pred_len, num_features
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_actors = num_actors
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_size = num_actors + num_features - 1
        self.hidden_dims = hidden_dims
        self.output_size = num_actors * pred_len
        self.name = "MLP"
        self.input_layer = nn.Linear(self.input_size * self.seq_len, hidden_dims[0])
        for i in range(len(hidden_dims) - 1):
            setattr(self, f"fc{i}", nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.output_layer = nn.Linear(hidden_dims[-1], self.output_size)
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            # Initialize linear layers with Kaiming initialization
            init.kaiming_uniform_(module.weight)
            if module.bias is not None:
                init.constant_(module.bias, 0)

    def forward(self, x):
        # Input dim is (batch_size, num_actors + num_temporal_features, seq_len)
        # Reshape to (batch_size, seq_len * num_actors * num_features)
        # Output dim is (batch_size, num_actors, pred_len) since we only predict HRV value
        p = 0.1
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.input_layer(x)
        x = F.relu(x)
        for i in range(len(self.hidden_dims) - 1):
            x = getattr(self, f"fc{i}")(x)
            x = F.relu(x)
            x = F.dropout(x, p=p)
        x = self.output_layer(x)
        x = x.view(batch_size, self.num_actors, self.pred_len)
        return x
