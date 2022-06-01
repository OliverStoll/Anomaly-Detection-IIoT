# Bachelor Thesis

The implementation of my federated learning approach for autoencoder-based condition monitoring of rotating machines.

The approach consists of a baseline of two different autoencoder models, one for raw vibration data and one for the computed fast-fourier-transformation (fft) of the data.
These autoencoder models are used to detect anomalies on the two different data sets provided, once fully centralized and once as a federated learning approach.

The goal of this is to evaluate the performance of federated learning on a relevant real-life use-case for the IIoT.

# How To Use

A full abstraction of the trainings- and evaluation process is provided in the `evaluation.py` file.
Here, either the comparison baseline, the centralized approach or the federated approach can be individually trained
and evaluated.
The training uses parameters specified in the `config.yaml` or `config_baseline.yaml` file.

# Project Structure

Both autoencoder models and the relevant functions for training and data preparation (normalization and batch-sizing) are implemented in `training.py`.
Some plotting functions are implemented in `plotting.py`, which are used to plot the most relevant metrics and some grafics about a models anomaly detection performance.

The federated learning implementation consists of two different worker scripts `worker_training.py` and `worker_aggregation.py`. 

These are containerized by their corresponding dockerfiles in `/configs`, producing two docker images that are ment to be run on a google compute engine testbed,
for evaluation of the quality of the federated learning approach.
