import tensorflow as tf
import os
import sys
import json
import keras_tuner as kt
from keras.callbacks import TensorBoard
from sklearn.preprocessing import StandardScaler
# from datetime import datetime
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import yaml
# from sklearn.metrics import roc_curve, auc
# import sklearn.metrics as skm
# from tensorboard.plugins.hparams import api as hp

from models import lstm_autoencoder_model, fft_autoencoder_model
from util.ml_calculations import *
from util.ml_callbacks import scheduler  # , tensor_callback
from util.config import c, config, client_config


class Training:
    def __init__(self, experiment_name, load_columns, train_columns, full_data_path=None, model_type="centralized"):
        # config values
        self.experiment_name = experiment_name
        self.load_columns = load_columns
        self.train_columns = train_columns
        self.split_size = c.SPLIT
        self.window_size = os.getenv('WINDOW_SIZE', c.WINDOW_SIZE)
        self.batch_size = c.BATCH_SIZE
        self.val_split = c.VAL_SPLIT
        self.train_split = c.TRAIN_SPLIT
        self.verbose = 1
        # data
        self.data_3d, self.data_train_3d = self._load_and_normalize_data(path=full_data_path)
        # self.fft_3d = fft_from_data(self.data_3d)
        # self.fft_train_3d = fft_from_data(self.data_train_3d)
        # self.fft_2d = self.fft_3d.reshape((-1, self.fft_3d.shape[2]))
        # models
        self.model_lstm = lstm_autoencoder_model()
        self.model_fft = fft_autoencoder_model()
        self.callbacks = [tf.keras.callbacks.LearningRateScheduler(scheduler)]  # initialize the learning rate scheduler
        # results
        self.history_lstm = {'loss': [], 'val_loss': []}
        self.history_fft = {'loss': [], 'val_loss': []}
        self.mses = {'lstm': [], 'fft': []}

        self.results = {'lstm': {}, 'fft': {}}
        self.logs_path = f"logs/{self.experiment_name}/centralized.json" if model_type == "centralized" \
            else f"logs/{self.experiment_name}/federated_{os.getenv('CLIENT_NAME')}.json"

        # print all attributes of the class
        print(os.getenv('CLIENT_NAME'))
        for attr, value in self.__dict__.items():
            print(f"{attr}: {value}") if ('2d' not in attr and '3d' not in attr) else print(f"{attr}: {value.shape}")

        print("\n")

    def _load_and_normalize_data(self, path=None):
        """ Prepare the data for training and evaluation, by normalizing and splitting the data """

        # read the data from the csv file
        if path is None:
            path = f"data/{self.experiment_name}/{self.split_size}.csv"
        if self.split_size == 20480 or self.split_size == 800:
            path = f"data/{self.experiment_name}/full.csv"  # for the full dataset
        print(f"Loading data from {path}")

        df = pd.read_csv(path, usecols=self.load_columns)

        # drop the last rows that are not a full split anymore
        if len(df) % self.split_size != 0:
            df = df.iloc[:-(len(df) % self.split_size)]

        # split percentage of data as training data, if specified as < 1
        split_len = int(len(df) * self.train_split) + (self.split_size - int(len(df) * self.train_split) % self.split_size)

        # split the data frame into multiple lists
        data = df.iloc[:, :].values
        train_data = df[:split_len].iloc[:, :].values
        train_data = train_data[:, self.train_columns]

        # normalize the data
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_data)

        # get all columns of the data
        data = data.T.reshape((-1, 1))
        train_data = train_data.T.reshape((-1, 1))

        data_windowed = [data[i:i + self.window_size] for i in range(0, len(data) - c.WINDOW_STEP, c.WINDOW_STEP)]
        data_windowed = np.concatenate(data_windowed, axis=0)
        data_train_windowed = [train_data[i:i + self.window_size] for i in range(0, len(train_data) - c.WINDOW_STEP, c.WINDOW_STEP)]
        data_train_windowed = np.concatenate(data_train_windowed, axis=0)

        # reshape data to 3d arrays with the second value equal to the number of timesteps (SPLIT)
        data_3d = data_windowed.reshape((-1, self.window_size, 1))
        train_data_3d = data_train_windowed.reshape((-1, self.window_size, 1))

        print(f"Data_3d shape: {data_3d.shape}")

        del df, data, data_windowed, data_train_windowed

        return data_3d, train_data_3d

    def load_models(self, dir_path):
        """ Load the models from the specified directory """

        # load the models
        self.model_lstm = tf.keras.models.load_model(f"{dir_path}/lstm.h5")
        self.model_fft = tf.keras.models.load_model(f"{dir_path}/fft.h5")

    def save_models(self, dir_path):
        """ Save the models to the specified directory"""

        # save the models
        self.model_lstm.save(f"{dir_path}/lstm.h5")
        self.model_fft.save(f"{dir_path}/fft.h5")

    def tune_models(self, replace_models=False):
        """ Hyperparameter tuning for the models, while logging the results """

        print("TUNING MODELS - REDIRECTING STDOUT")

        # save stdout to a file
        old_stdout = sys.stdout
        # delete the old log file and create folder if not existing
        log_path = f"hyper_tuning/tuning_log.txt"
        os.makedirs("hyper_tuning", exist_ok=True)
        if os.path.exists(log_path):
            os.remove(log_path)
        sys.stdout = open(f"hyper_tuning/tuning_log.txt", "w")

        tb_lstm = TensorBoard(log_dir=f"hyper_tuning/logs/lstm")
        tb_fft = TensorBoard(log_dir=f"hyper_tuning/logs/fft")
        tuner_lstm = kt.RandomSearch(lstm_autoencoder_model, objective='val_loss',
                                     project_name="hyper_tuning/lstm", max_trials=1000000)
        tuner_fft = kt.RandomSearch(fft_autoencoder_model, objective='val_loss',
                                    project_name="hyper_tuning/fft", max_trials=1000000)
        tuner_lstm.search(self.data_train_3d,
                          self.data_train_3d,
                          epochs=50,
                          batch_size=self.batch_size,
                          validation_split=self.val_split,
                          callbacks=[tb_lstm],
                          verbose=2)
        tuner_fft.search(self.fft_train_3d,
                         self.fft_train_3d,
                         epochs=50,
                         batch_size=self.batch_size,
                         validation_split=self.val_split,
                         callbacks=[tb_fft])
        tuner_lstm.results_summary(num_trials=1)
        print()
        tuner_fft.results_summary(num_trials=1)

        # restore stdout
        sys.stdout = old_stdout

        if replace_models:
            self.model_lstm = tuner_lstm.get_best_models(num_models=1)[0]
            self.model_fft = tuner_fft.get_best_models(num_models=1)[0]

    def train_models(self, epochs=1):
        """ Train the model for a given number of epochs. """

        # starting time
        start = datetime.now()
        print(f"TRAINING MODELS with verbose: {self.verbose}")
        print(f"TRAINING-START {start}")

        train_dict = {'epochs': epochs, 'batch_size': self.batch_size, 'callbacks':self.callbacks,
                      'validation_split': self.val_split, 'verbose': self.verbose}

        # train the models
        _history_lstm = self.model_lstm.fit(self.data_train_3d, self.data_train_3d, **train_dict).history
        self.history_lstm['loss'] += _history_lstm['loss']
        if self.val_split:
            self.history_lstm['val_loss'] += _history_lstm['val_loss']

        mid = datetime.now()
        print(f"TRAINING-MID {mid} - LSTM: {(mid - start).total_seconds()}")

        if os.getenv("TRAINING_FFT", "False") == "True":
            _history_fft = self.model_fft.fit(self.fft_train_3d, self.fft_train_3d, **train_dict).history
            self.history_fft['loss'] += _history_fft['loss']
            if self.val_split:
                self.history_fft['val_loss'] += _history_fft['val_loss']

        self.results['lstm']['loss'] = self.history_lstm['loss']

        print(f"TRAINING-DONE {datetime.now()}")
        print(f"TRAINING-TIME: {(datetime.now() - start).total_seconds()}")

    def calculate_reconstruction_error(self):
        """ Calculate the anomaly score for the data. """

        # calculate the anomaly scores
        print("CALCULATING RE-SCORE")
        data_lstm_3d = self.data_3d[::2, :, :]  # accounting the overlap
        data_lstm_2d = data_lstm_3d.reshape((-1, data_lstm_3d.shape[2]))
        self.pred_2d_lstm = self.model_lstm.predict(data_lstm_3d, verbose=1).reshape((-1, data_lstm_3d.shape[2]))
        self.mses = {'lstm': ((data_lstm_2d - self.pred_2d_lstm) ** 2).mean(axis=1)}
        # calculate the mean of 1000 mses
        self.mses['lstm'] = self.mses['lstm'].reshape(-1, 1000).mean(axis=1)

        # save results
        self.results['lstm']['mse'] = self.mses['lstm'].tolist()

    def save_results(self, save_path=None):
        """ Save the results to a file. """

        if not self.mses['lstm']:
            self.calculate_reconstruction_error()

        save_path = self.logs_path if save_path is None else save_path

        # take the logs path without the file name
        dir_path = save_path[:-len(save_path.split('/')[-1])]
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # save the results
        with open(save_path, 'w') as f:
            json.dump(self.results, f)
        print(f"SAVED RESULTS TO {save_path}")
        # save the predictions to a json file
        try:
            save_predictions_path = save_path.replace('.json', '_predictions.json')
            with open(save_predictions_path, 'w') as f:
                json.dump({'Prediction': self.pred_2d_lstm.tolist()}, f)
        except:
            print("Could not save the predictions")


if __name__ == '__main__':

    # EVALUATE TRAINING AS DOCKER IMAGE
    for path, subdirs, files in os.walk("./data"):
        for name in files:
            print(os.path.join(path, name))

    experiments = [2]
    epochs = int(os.getenv("EPOCHS", c.EPOCHS))
    for experiment in experiments:
        experiment_name = f"bearing_experiment-{experiment}"
        columns = config['LOAD_COLUMNS_DICT'][experiment_name]
        trainer = Training(experiment_name=experiment_name, train_columns=[0, 1, 2, 3],
                           load_columns=columns)
        trainer.model_lstm.summary()
        trainer.train_models(epochs=epochs)
        trainer.save_results()