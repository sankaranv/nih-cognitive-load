from utils.config import config
from utils.create_batches import create_torch_loader_from_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.stats import compute_regression_metrics
import numpy as np
import os


class ConditionalVAE(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.num_actors = len(config.role_names)
        self.seq_len = model_config["seq_len"]
        self.pred_len = model_config["pred_len"]
        self.model_name = model_config["model_name"]
        self.encoder_dims = model_config["encoder_dims"]
        self.decoder_dims = model_config["decoder_dims"]
        self.latent_dim = model_config["latent_dim"]
        self.num_epochs = model_config["num_epochs"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # For the encoder we concatenate the current timestep and the previous seq_len timesteps
        # We ignore temporal features for input since we don't need to reconstruct them
        self.input_size = (
            self.num_actors * (self.seq_len + self.pred_len) + len(config.phases) + 2
        )
        # For the decoder we only use the current timestep and add previous timesteps as input
        self.condition_size = self.num_actors * self.seq_len + len(config.phases) + 2
        self.output_size = self.num_actors * self.pred_len

        # Encoder
        self.encoder_input = nn.Linear(self.input_size, self.encoder_dims[0])
        for i in range(len(self.encoder_dims) - 1):
            setattr(
                self,
                f"fc_encoder{i}",
                nn.Linear(self.encoder_dims[i], self.encoder_dims[i + 1]),
            )
        self.latent_mu = nn.Linear(self.encoder_dims[-1], self.latent_dim)
        self.latent_logvar = nn.Linear(self.encoder_dims[-1], self.latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(
            self.latent_dim + self.condition_size, self.decoder_dims[0]
        )
        for i in range(len(self.decoder_dims) - 1):
            setattr(
                self,
                f"fc_decoder{i}",
                nn.Linear(self.decoder_dims[i], self.decoder_dims[i + 1]),
            )
        self.decoder_output = nn.Linear(self.decoder_dims[-1], self.output_size)

    def encoder(self, x, history):
        x = torch.cat((x, history), dim=1)
        x = self.encoder_input(x)
        x = torch.relu(x)
        for i in range(len(self.encoder_dims) - 1):
            x = getattr(self, f"fc_encoder{i}")(x)
            x = torch.relu(x)
        mu = self.latent_mu(x)
        logvar = self.latent_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decoder(self, z, history):
        x = torch.cat((z, history), dim=1)
        x = self.decoder_input(x)
        x = torch.relu(x)
        for i in range(len(self.decoder_dims) - 1):
            x = getattr(self, f"fc_decoder{i}")(x)
            x = torch.relu(x)
        return self.decoder_output(x)

    def forward(self, x, history):
        # x dim is (batch_size, num_features, pred_len) where num_features = num_actors + num_temporal_features
        # Encoder input x should be (batch_size, num_actors * self.pred_len)
        batch_size = x.shape[0]
        num_actors = config.num_actors
        hrv_in = x[:, 0:num_actors, :].reshape(batch_size, -1)
        temporal_in = x[:, num_actors:, -1].reshape(batch_size, -1)
        x = torch.cat((hrv_in, temporal_in), dim=1)

        # history is (batch_size, num_features, seq_len) where num_features = num_actors + num_temporal_features
        # Encoder input history should be (batch_size, num_actors * self.seq_len)
        # We use temporal features from the last timestep
        hrv_history = history[:, 0:num_actors, :].reshape(batch_size, -1)
        temporal_history = history[:, num_actors:, -1].reshape(batch_size, -1)
        history = torch.cat((hrv_history, temporal_history), dim=1)

        mu, logvar = self.encoder(x, history)
        z = self.reparameterize(mu, logvar)
        out = self.decoder(z, history)

        # Reshape output to (batch_size, num_actors, pred_len)
        out = out.view(batch_size, num_actors, self.pred_len)
        return out, mu, logvar


class ConditionalVAEAnomalyDetector:
    def __init__(self, model_config):
        self.base_model = model_config["base_model"]
        self.model_name = model_config["model_name"]
        self.seq_len = model_config["seq_len"]
        self.pred_len = model_config["pred_len"]
        self.num_epochs = model_config["num_epochs"]
        self.model_config = model_config
        self.setup_models()

    def setup_models(self):
        self.models = {}
        self.optimizer = {}
        # self.criterion = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for param in config.param_names:
            self.models[param] = ConditionalVAE(self.model_config).to(self.device)
            self.optimizer[param] = torch.optim.Adam(
                self.models[param].parameters(), lr=self.model_config["lr"]
            )

    def save(self, model_dir):
        if not os.path.exists(f"{model_dir}/{self.model_name}"):
            os.makedirs(f"{model_dir}/{self.model_name}")
        for param in config.param_names:
            torch.save(
                self.models[param].state_dict(),
                f"{model_dir}/{self.model_name}/{self.model_name}_{param}.pt",
            )

    def criterion(self, output, target, mu, logvar):
        # Reconstruction loss
        recon_loss = F.mse_loss(output, target, reduction="sum")
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss

    def train(self, train_dataset, val_dataset=None, verbose=False):
        # Train joint models for each parameter
        trace = {}
        for param in config.param_names:
            print(f"Training {self.model_name} for {param}")
            train_loader = create_torch_loader_from_dataset(
                train_dataset,
                self.seq_len,
                self.pred_len,
                param,
                self.model_config["batch_size"],
                shuffle=True,
            )
            if val_dataset is not None:
                val_loader = create_torch_loader_from_dataset(
                    val_dataset,
                    self.seq_len,
                    self.pred_len,
                    param,
                    self.model_config["batch_size"],
                    shuffle=False,
                )
            else:
                val_loader = None

            # Train each model
            trace[param] = {"train_loss": [], "val_loss": []}
            for epoch in range(self.num_epochs):
                self.models[param].train()
                train_loss = 0
                for batch_idx, (history, target) in enumerate(train_loader):
                    history, target = history.to(self.device), target.to(self.device)
                    self.optimizer[param].zero_grad()
                    output, mu, logvar = self.models[param].forward(target, history)
                    loss = self.criterion(output, target, mu, logvar)
                    loss.backward()
                    self.optimizer[param].step()
                    train_loss += loss.item()
                train_loss /= len(train_loader.dataset)
                trace[param]["train_loss"].append(train_loss)

                # Validation
                if val_loader is not None:
                    self.models[param].eval()
                    val_loss = 0
                    with torch.no_grad():
                        for history, target in val_loader:
                            history, target = history.to(self.device), target.to(
                                self.device
                            )
                            output, mu, logvar = self.models[param].forward(
                                target, history
                            )
                            val_loss += self.criterion(
                                output, target, mu, logvar
                            ).item()
                    val_loss /= len(val_loader.dataset)
                    trace[param]["val_loss"].append(val_loss)
                    if verbose:
                        print(
                            f"Epoch {epoch + 1}: Train Loss: {train_loss}, Val Loss: {val_loss}"
                        )
                else:
                    if verbose:
                        print(f"Epoch {epoch + 1}: Train Loss: {train_loss}")

        return trace

    def predict_proba(self, dataset):
        # Objects for collecting probabilities
        probs = {}
        for case_idx, case_data in dataset.items():
            probs[case_idx] = {}
            num_samples = case_data.shape[-1]
            for param in config.param_names:
                probs[case_idx][param] = np.zeros(
                    (config.num_actors, num_samples - self.seq_len)
                )

        trace = {param: {"test_loss": None} for param in config.param_names}
        for param in config.param_names:
            # Create dataloader
            param_idx = config.param_indices[param]
            test_loader = create_torch_loader_from_dataset(
                dataset,
                self.seq_len,
                self.pred_len,
                param,
                self.model_config["batch_size"],
                shuffle=False,
            )

            # Obtain probabilities
            self.models[param].eval()
            test_loss = 0
            num_samples = 0
            probs_list_per_param = []

            with torch.no_grad():
                for history, target in test_loader:
                    # Obtain reconstructed outputs
                    history, target = history.to(self.device), target.to(self.device)
                    output, mu, logvar = self.models[param].forward(target, history)

                    # Compute test loss
                    test_loss += self.criterion(output, target, mu, logvar).item()

                    # Get negative log likelihood of each target
                    num_samples += history.shape[0]
                    log_probs = F.mse_loss(output, target, reduction="none")
                    log_probs = log_probs.view(-1).cpu()
                    print(log_probs.shape)
                    probs_list_per_param.append(log_probs.cpu().numpy())

            test_loss /= len(test_loader.dataset)
            trace[param]["test_loss"] = test_loss

            # Split probabilities into individual cases
            idx = 0
            for case_idx, case_data in dataset.items():
                num_samples = case_data.shape[-1]
                probs[case_idx][param] = np.array(probs_list_per_param)[
                    :, idx : idx + num_samples - self.seq_len
                ]
                idx += num_samples - self.seq_len

        return probs, trace
