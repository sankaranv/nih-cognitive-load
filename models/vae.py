import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import config
from utils.create_batches import create_torch_loader_from_dataset
import utils.training as train_utils
import numpy as np


class VAE(nn.Module):
    def __init__(self, model_config):
        super(VAE, self).__init__()
        self.input_dim = config.num_actors + config.num_phases + 2
        self.hidden_dim = model_config["hidden_dim"]

        # Encoder layers
        self.fc1 = nn.Linear(self.input_dim, 512)
        self.fc21 = nn.Linear(512, self.hidden_dim)  # Mean of the hidden space
        self.fc22 = nn.Linear(512, self.hidden_dim)  # Log variance of the hidden space

        # Decoder layers
        self.fc3 = nn.Linear(self.hidden_dim, 512)
        self.fc4 = nn.Linear(512, self.input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)  # Mean and log variance

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VAEAnomalyDetector:
    def __init__(self, model_config):
        self.base_model = model_config["base_model"]
        self.model_name = model_config["model_name"]
        self.seq_len = model_config["seq_len"]
        self.num_epochs = model_config["num_epochs"]
        self.model_config = model_config
        self.pred_len = 1
        self.setup_models()

    def setup_models(self):
        self.models = {}
        self.optimizer = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for param in config.param_names:
            self.models[param] = VAE(self.model_config).to(self.device)
            self.optimizer[param] = torch.optim.Adam(
                self.models[param].parameters(), lr=self.model_config["lr"]
            )

    def vae_loss(self, recon_x, x, mu, logvar, param):
        BCE = F.binary_cross_entropy(
            recon_x, x.view(-1, self.models[param].input_dim), reduction="sum"
        )
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    def train(self, train_dataset, val_dataset=None, verbose=False):
        # Train joint models for each parameter
        trace = {
            param: {"train_loss": [], "val_loss": []} for param in config.param_names
        }
        for param in config.param_names:
            if param == "LF-HF":
                continue
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

            # Print shape of input and output
            for epoch in range(self.num_epochs):
                self.models[param].train()
                train_loss = 0
                for batch_idx, (data, _) in enumerate(train_loader):
                    # Get data for the current parameter
                    # Currently the size of the data is (batch_size, num_features, num_params)
                    data = data[:, :, config.param_indices[param]]
                    data = data.to(self.device)
                    # Train model
                    self.optimizer[param].zero_grad()
                    recon_batch, mu, logvar = self.models[param](data)
                    loss = self.vae_loss(recon_batch, data, mu, logvar, param)
                    loss.backward()
                    self.optimizer[param].step()
                    train_loss += loss.item()
                train_loss /= len(train_loader.dataset)
                trace[param]["train_loss"].append(train_loss)
                if val_dataset is not None:
                    self.models[param].eval()
                    val_loss = 0
                    with torch.no_grad():
                        for data, _ in val_loader:
                            data = data.to(self.device)
                            recon_batch, mu, logvar = self.models[param](data)
                            val_loss += self.vae_loss(
                                recon_batch, data, mu, logvar, param
                            )
                    val_loss /= len(val_loader.dataset)
                    trace[param]["val_loss"].append(val_loss)
                if verbose:
                    if val_dataset is not None:
                        print(
                            f"Epoch {epoch + 1}: Train Loss: {train_loss}, Val Loss: {val_loss}"
                        )
                    else:
                        print(f"Epoch {epoch + 1}: Train Loss: {train_loss}")
        return trace

    def predict_proba(self, test_dataset, trace=None):
        """
        Predict probabilities for each point in the dataset
        Args:
            test_dataset:
            trace:

        Returns:

        """
        # if trace is None or len(trace) == 0:
        #     trace = {param: {} for param in config.param_names}
        case_ids = list(test_dataset.keys())
        log_probs = {param: [] for param in config.param_names}

        probs = {}
        for case_idx, case_data in test_dataset.items():
            probs[case_idx] = {}
            num_samples = case_data.shape[-1]
            for param in config.param_names:
                probs[case_idx][param] = np.zeros(
                    (config.num_actors, num_samples - self.seq_len)
                )

        for param in config.param_names:
            if param == "LF-HF":
                continue
            param_idx = config.param_indices[param]
            test_loader = create_torch_loader_from_dataset(
                test_dataset,
                self.seq_len,
                self.pred_len,
                param,
                batch_size=1,
                shuffle=False,
            )
            for batch_idx, (data, _) in enumerate(test_loader):
                # Get data for the current parameter
                data = data[:, :, config.param_indices[param]]
                data = data.to(self.device)
                self.models[param].eval()
                recon_batch, mu, logvar = self.models[param](data)
                log_prob = -self.vae_loss(recon_batch, data, mu, logvar, param)
                log_probs[param].append(log_prob.item())

            # Split the log_probs into cases
            idx = 0
            for case_idx in case_ids:
                num_samples = test_dataset[case_idx].shape[-1]
                offset = self.seq_len + self.pred_len - 1
                case_predictions = log_probs[param][idx : idx + num_samples - offset]
                for actor_idx in range(config.num_actors - 1):
                    probs[case_idx][param][actor_idx] = case_predictions
            print(probs.keys())
        return probs
