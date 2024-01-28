from models.joint_linear import JointLinearModel
from models.param_free import ParameterFreeAutoregressiveModel
from utils.data import load_dataset
from utils.training import cross_validation
from visualization.all_plots import *

model_config = {
    "model_name": "ParameterFreeAutoregressive",
    "seq_len": 5,
    "pred_len": 1,
}

dataset, unimputed_dataset = load_dataset(
    data_dir="./data/processed", normalized=True, pad_phase_on=False
)

param = "PNS index"
# model = ParameterFreeAutoregressiveModel(model_config)
#
# model.train(dataset, verbose=True)
# trace = model.predict(dataset)

model, trace = cross_validation(
    "ParameterFreeAutoregressive",
    model_config,
    dataset,
    num_folds=5,
    verbose=True,
)

print("Plotting predictions")
plot_predictions(
    model.model_name,
    trace,
    dataset,
    unimputed_dataset,
    model.seq_len,
    model.pred_len,
    "./plots/5min",
)

print("Plotting correlation densities")
eval_metric_density_plots(model.model_name, trace, dataset, "corr_coef", "./plots/5min")

print("Plotting scatterplots")
generate_scatterplots(model, trace, dataset, "./plots/5min")

print("Plotting feature importances")
plot_feature_importances(model, model.seq_len, "./plots/5min")
