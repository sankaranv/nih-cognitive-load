from models import *
from visualization.numpy_plots import *


def linear_experiments(model, train_dataset, test_dataset, unimputed_test_dataset, plots_dir, fold):

    print(f"Training {model.name} model")
    model.train(train_dataset, model_dir="./checkpoints/ridge")
    predictions = model.predict(test_dataset)
    print("Generating plots for predictions")

    plot_predictions(
        model=model,
        predictions=predictions,
        test_dataset=test_dataset,
        unimputed_test_dataset=unimputed_test_dataset,
        plots_dir=f"{plots_dir}/5min/predictions/ridge",
    )

    print("Generating scatterplots")
    generate_scatterplots(model=model,
                          predictions=predictions,
                          test_dataset=unimputed_test_dataset,
                          plots_dir=f"{plots_dir}/5min/scatterplots/{model.name}")

    print("Generating density plots")
    correlation_density_plots(model=model,
                              predictions=predictions,
                              test_dataset=unimputed_test_dataset,
                              plots_dir=f"{plots_dir}/5min/density_plots/{model.name}/fold_{fold}/")

    print(f"{model.name} experiments complete!")

    print("Generating plots for feature importances")
    plot_feature_importances(
        model,
        param_names,
        role_names,
        plots_dir=f"{plots_dir}/5min/predictions/{model.name}",
    )


def param_free_autoregressive_experiments(test_dataset, unimputed_test_dataset, plots_dir):

    print("Training parameter-free autoregressive model")
    model = ParameterFreeAutoregressiveModel(
            param_names=param_names,
            role_names=role_names)

    # No training, just predictions
    predictions = model.predict(test_dataset)
    print("Generating plots for predictions")
    plot_predictions(
            model=model,
            predictions=predictions,
            test_dataset=test_dataset,
            unimputed_test_dataset=unimputed_test_dataset,
            plots_dir=f"{plots_dir}/5min/predictions/parameter_free",
        )
    print("Generating scatterplots")
    generate_scatterplots(model=model,
                          predictions=predictions,
                          test_dataset=unimputed_test_dataset,
                          plots_dir=f"{plots_dir}/5min/scatterplots/parameter_free")

    print("Generating density plots")
    correlation_density_plots(model=model,
                              predictions=predictions,
                              test_dataset=unimputed_test_dataset,
                              plots_dir=f"{plots_dir}/5min/density_plots/{model.name}")

    print("Parameter-free autoregressive model experiments done!")
