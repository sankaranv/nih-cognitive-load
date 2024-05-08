# Analysis and Prediction of Cognitive Load among Team Members during Cardiac Surgery

This repo includes imputation models, time-series prediction models, and anomaly detection models.

`python hrv_prediction.py --model model_config_name`
`python anomaly_detection.py --model model_config_name`
`python impute_data.py`

See the scripts for additional options. The model configs are written in JSON format and can be found in the `experiments/` folder; additional experiments can be run by writing JSON files in the same format. 

### Time-series Prediction
- Transformer with continuous-valued inputs
- LSTM
- Gaussian Processes with RBF, Rational Quadratic, and Matern kernels
- Ridge Regression
- MLPs
- Parameter Free Autoregressive Model
- Random Baseline
- XGBoost

### Anomaly Detection
- Conditional Variational Autoencoder (CVAE) with MLPs as encoder and decoder.
- Gaussian Processes with RBF kernel. Any `GPy` kernel can be used to replace the RBF kernel.
- Dependency Networks with Random Forests as the base model. Any `sklearn` classifier that includes a `predict_proba` function can be used as a base model.
