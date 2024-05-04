from models.gaussian_process import VariationalGP
from utils.data import load_dataset
from utils.training import cross_validation, create_model

model_config = {
    "base_model": "VariationalGP",
    "model_name": "VariationalGP",
    "seq_len": 5,
    "device": "cpu",
    "num_epochs": 500,
    "lr": 1e-2,
}
model_type = "VariationalGP"

dataset, unimputed_dataset = load_dataset(
    data_dir="./data/processed",
    normalized=True,
    pad_phase_on=True,
)

model, trace = cross_validation(
    model_type,
    model_config,
    dataset,
    num_folds=5,
    verbose=True,
)
