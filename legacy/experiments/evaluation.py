from utils.data import cv_split
from experiments.mlp import mlp_experiment
from utils.globals import param_names


def cross_validate_experiment(
    model,
    model_name,
    dataset,
    unimputed_dataset,
    plots_dir,
    model_dir,
    num_folds=5,
    **kwargs,
):
    all_predictions = {param: {} for param in param_names}

    cv_splits = cv_split(dataset, num_folds=num_folds)
    for i, (train_dataset, val_dataset, test_dataset) in enumerate(cv_splits):
        print(f"Running CV split {i+1}")
        print(f"Train cases: {train_dataset.keys()}")
        print(f"Val cases: {val_dataset.keys()}")
        print(f"Test cases: {test_dataset.keys()}")

        # Apply same split to original dataset
        unimputed_train_dataset = {
            key: unimputed_dataset[key] for key in train_dataset.keys()
        }
        unimputed_val_dataset = {
            key: unimputed_dataset[key] for key in val_dataset.keys()
        }
        unimputed_test_dataset = {
            key: unimputed_dataset[key] for key in test_dataset.keys()
        }

        # Use dataset to get number of features
        key = list(dataset.keys())[0]
        num_features = dataset[key].shape[-2]

        if model.__class__.__name__ in ["RidgeRegressionModel", "XGBRegressionModel"]:
            pass
        elif model.__class__.__name__ == "ParameterFreeAutoregressiveModel":
            pass
        elif model.__class__.__name__ == "MLP":
            # Train model and get traces
            trace = mlp_experiment(
                kwargs["hidden_dims"],
                train_dataset,
                val_dataset,
                test_dataset,
                unimputed_test_dataset,
                plots_dir,
                model_dir,
                kwargs["device"],
                kwargs["batch_size"],
                kwargs["num_epochs"],
                kwargs["seq_len"],
                kwargs["pred_len"],
                kwargs["learning_rate"],
            )

            # Add predictions to overall set of predictions
            for param in param_names:
                all_predictions[param].update(trace["predictions"][param])

        else:
            raise ValueError(f"Unknown model {model.__class__.__name__}")

    return all_predictions
