import math
import torch
from torch import nn, Tensor
import torch.nn.init as init
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer


class PositionalEncoding(nn.Module):
    """Encodes absolute position of elements in the sequence of HRV measurements.
        Inputs have shape (batch_size, d_model)
        This is given using sin and cos functions of the inputs so there is no need to learn the position embeddings.
    Args:
        d_model (int): Dimension of the model.
        max_len (int): Maximum length of the sequence.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
    """

    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        encoding = position * div_term
        self.pe = torch.zeros(max_len, d_model)
        self.pe[:, 0::2] = torch.sin(encoding)
        self.pe[:, 1::2] = torch.cos(encoding)

    def forward(self, x: Tensor, time_indices) -> Tensor:
        """Add positional embedding to the input tensor.

        Args:
            x (Tensor): Input tensor of shape (batch_size, d_model, seq_len).
            time_indices (Tensor): Tensor of shape (batch_size, seq_len) containing the time indices of the input.

        Returns:
            Tensor: Output tensor of shape (batch_size, d_model, seq_len).
        """
        time_vector = time_indices.to(dtype=torch.int64)
        # enc_with_time = torch.zeros(batch_size, d_model, seq_len)
        # for i in range(time_vector.shape[0]):
        #     enc_with_time[i, :, :] = self.pe[time_vector[i],:].transpose(0,1)
        enc_with_time = self.pe[time_vector.unsqueeze(1), :].squeeze().transpose(1,-1)
        return enc_with_time + x




class PhaseEncoding(nn.Module):
    """Embeds phase ID as a vector of size d_model using a token-vocabulary lookup table.
    Embeddings are summed for each phase that is active
    An extra embedding is added to represent the absence of a phase.
    Inputs have shape (batch_size, num_features - 2, seq_len)"""

    def __init__(self, d_model: int, n_phases: int, dropout: float = 0.1):
        super().__init__()
        self.n_phases = n_phases
        self.embeddings = nn.Embedding(n_phases + 1, d_model)
        self.dropout = nn.Dropout(p=dropout)
        init.xavier_uniform_(self.embeddings.weight)

    def forward(self, x, phase_one_hot: Tensor) -> Tensor:
        """For each of the phases that are active, add the corresponding embedding to the input tensor.
        One-hot encodings are of shape (batch_size, n_phases + 1, seq_len)
        """
        batch_size, num_features, seq_len = phase_one_hot.shape
        d_model = x.shape[1]
        phase_one_hot_reshaped = phase_one_hot.permute(0,2,1).unsqueeze(3)
        emb = phase_one_hot_reshaped * self.embeddings.weight.view(1, 1, num_features, d_model)
        x += emb.sum(dim=2).permute(0, 2, 1)
        return self.dropout(x)


class ContinuousTransformer(nn.Module):
    """Create a transformer with continuous inputs and outputs.
        Token embeddings are replaced with a linear layer since we have real valued inputs.
        Input dimensions are (batch_size, num_features, seq_len)
    Args:
        d_model (int): Dimension of the model.
        n_heads (int): Number of attention heads.
        d_hidden (int): Dimension of the hidden layer.
        n_layers (int): Number of layers.
        n_features (int): Number of features in the input for multi-dimensional time series.
        dropout (float, optional): Dropout probability. Defaults to 0.5.
    """

    def __init__(
        self, architecture, seq_len, pred_len, max_len, num_actors
    ):
        super().__init__()
        self.d_model = architecture["d_model"]
        self.n_heads = architecture["n_heads"]
        self.d_hidden = architecture["d_hidden"]
        self.n_enc_layers = architecture["n_enc_layers"]
        self.n_dec_layers = architecture["n_dec_layers"]
        self.dropout = architecture["dropout"]
        self.n_phases = architecture["n_phases"]
        self.max_len = max_len
        self.arch = architecture
        self.name = "ContinuousTransformer"

        self.hrv_encoder = nn.Linear(num_actors, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, self.max_len, self.dropout)
        self.phase_encoder = PhaseEncoding(self.d_model, self.n_phases, self.dropout)
        encoder_layers = TransformerEncoderLayer(
            self.d_model, self.n_heads, self.d_hidden, self.dropout
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, self.n_enc_layers)
        self.target_decoder = nn.Linear(num_actors * pred_len, self.d_model)
        decoder_layers = TransformerDecoderLayer(self.d_model, self.n_heads, self.d_hidden, self.dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, self.n_dec_layers)
        self.output_layer = nn.Linear(self.d_model * seq_len, num_actors * pred_len)
        # self._init_weights()

        self.num_actors = num_actors
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.apply(self.initialize_weights)

    def initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            # Initialize linear layers with Xavier/Glorot initialization
            init.xavier_uniform_(module.weight)
            if module.bias is not None:
                init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            # Initialize layer normalization layers
            init.constant_(module.weight, 1)
            init.constant_(module.bias, 0)
        elif isinstance(module, nn.MultiheadAttention):
            # Initialize attention layers
            init.xavier_uniform_(module.in_proj_weight)
            init.xavier_uniform_(module.out_proj.weight)
            if module.in_proj_bias is not None:
                init.constant_(module.in_proj_bias, 0)
            if module.out_proj.bias is not None:
                init.constant_(module.out_proj.bias, 0)

    def forward(self, x):
        # Input dim is (batch_size, num_actors + num_temporal_features, seq_len)
        # Reshape to (batch_size, seq_len * num_actors * num_features)
        # Output dim is (batch_size, num_actors, pred_len) since we only predict HRV value

        # Use only the first four dims in encoder to get shape (batch_size, num_actors, seq_len)
        hrv_in = x[:, :self.num_actors, :]
        # Reshape to (batch_size * seq_len, num_actors)
        batch_size = hrv_in.shape[0]
        seq_len = hrv_in.shape[-1]
        hrv_in = hrv_in.transpose(1, 2).reshape(-1, self.num_actors)
        out = self.hrv_encoder(hrv_in) * math.sqrt(self.d_model)
        # Output shape is (batch_size * seq_len, d_model)
        # Reshape to (batch_size, d_model, seq_len)
        out = out.reshape(batch_size, self.d_model, seq_len)
        # Use last dim for positional encoding to get shape (batch_size, seq_len)
        time_idx = x[:, -1, :]
        out = self.pos_encoder(out, time_idx)

        # Use remaining dims for phase encoding to get shape (batch_size, num_features - 2, seq_len)
        phase_one_hot = x[:, self.num_actors:-1, :]
        out = self.phase_encoder(out, phase_one_hot)

        # Reshape to (batch_size, seq_len, d_model) for transformer encoder
        out = out.permute(0, 2, 1)
        out = self.transformer_encoder(out)

        # Not using transformer decoder but just linear decoder
        out = out.reshape(batch_size, -1)
        out = self.output_layer(out)
        out = out.reshape(batch_size, self.num_actors, self.pred_len)

        return out

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()