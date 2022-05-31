import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import keras_tuner as kt
from keras.callbacks import TensorBoard
from tensorboard.plugins.hparams import api as hp
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from io import StringIO

from plotting import *  # evaluate_model_lstm, evaluate_model_fft, find_timestretches_from_indexes
from models import lstm_autoencoder_model, fft_autoencoder_model
from util.calculations import *
from util.callbacks import scheduler  # , tensor_callback
from util.config import c, c_client

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # disable GPU usage (IIoT)


class Training:
    def __init__(self, data_path, data_columns):
        # config values
        self.data_path = data_path
        self.dataset_name = data_path.split('/')[-2]
        self.experiment_name = data_path.split('/')[-1]
        self.sub_experiment_index = int(data_columns[0] / len(data_columns))
        self.data_columns = data_columns
        self.split_size = c.SPLIT
        self.batch_size = c.BATCH_SIZE
        self.val_split = c.VAL_SPLIT
        self.train_split = c.TRAIN_SPLIT
        # data
        self.labels = anomalies[self.dataset_name][self.experiment_name]['bearing-0']  # todo: generalize
        self.data_3d, self.data_train_3d = self._load_and_normalize_data()
        self.data_2d = self.data_3d.reshape((-1, self.data_3d.shape[2]))
        self.fft_3d = calculate_fft_from_data(self.data_3d)
        self.fft_train_3d = calculate_fft_from_data(self.data_train_3d)
        self.fft_2d = self.fft_3d.reshape((-1, self.fft_3d.shape[2]))
        # models and losses
        self.model_lstm = lstm_autoencoder_model()
        self.model_fft = fft_autoencoder_model()
        self.history_lstm = {'loss': [], 'val_loss': []}
        self.history_fft = {'loss': [], 'val_loss': []}
        self.mse_lstm = []
        self.mse_fft = []
        self.data_pred_2d = []
        self.fft_pred_2d = []
        self.callbacks = [tf.keras.callbacks.LearningRateScheduler(scheduler),
                          TensorBoard(log_dir=f"logs/train/lstm")]  # initialize the learning rate scheduler

    def _load_and_normalize_data(self):
        """
        Prepare the data for training and evaluation. All features need to be extracted and named.

        The data is normalized and possibly split into training and testing data 80/20.

        :return: the full data, the training data and the 3d array of the training data
        """

        # read the data from the csv file
        path = f"{self.data_path}_{self.split_size}.csv"
        if self.split_size == 20480 or self.split_size == 800:
            path = f"{self.data_path}_full.csv"  # for the full dataset

        df = pd.read_csv(path, usecols=self.data_columns)

        # drop the last rows that are not a full split anymore
        if len(df) % self.split_size != 0:
            df = df.iloc[:-(len(df) % self.split_size)]

        # split percentage of data as training data, if specified as < 1
        split_len = int(len(df) * self.train_split) \
                    + (self.split_size - int(len(df) * self.train_split) % self.split_size)

        # split the data frame into multiple lists
        data = df.iloc[:, :].values
        train_data = df[:split_len].iloc[:, :].values

        # normalize the data
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
        train_data = scaler.transform(train_data)

        # reshape data to 3d arrays with the second value equal to the number of timesteps (SPLIT)
        data_3d = data.reshape((-1, self.split_size, data.shape[1]))
        train_data_3d = train_data.reshape((-1, self.split_size, train_data.shape[1]))

        return data_3d, train_data_3d

    def load_models(self, dir_path):
        """
        Load the models from the specified directory.

        :param dir_path: the directory path to the models
        :return: the model and the fft model
        """

        # load the models
        self.model_lstm = tf.keras.models.load_model(f"{dir_path}/lstm_model.h5")
        self.model_fft = tf.keras.models.load_model(f"{dir_path}/fft_model.h5")

    def save_models(self, dir_path):
        """
        Save the models to the specified directory.

        :param dir_path: the directory path to the models
        """

        # save the models
        self.model_lstm.save(f"{dir_path}/lstm_model.h5")
        self.model_fft.save(f"{dir_path}/fft_model.h5")

    def tune_models(self, replace_models=False):
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
        """
        Train the model for a given number of epochs. The training data needs to be formatted as a 3D array.

        :param epochs: the number of epochs to train for
        :return: the trained model
        """

        # train the models
        _history_lstm = self.model_lstm.fit(self.data_train_3d,
                                            self.data_train_3d,
                                            epochs=epochs,
                                            batch_size=self.batch_size,
                                            callbacks=self.callbacks,
                                            validation_split=self.val_split).history
        _history_fft = self.model_fft.fit(self.fft_train_3d,
                                          self.fft_train_3d,
                                          epochs=epochs,
                                          batch_size=self.batch_size,
                                          validation_split=self.val_split).history

        self.history_lstm['loss'] += _history_lstm['loss']
        self.history_lstm['val_loss'] += _history_lstm['val_loss']
        self.history_fft['loss'] += _history_fft['loss']
        self.history_fft['val_loss'] += _history_fft['val_loss']

    def evaluate(self, show_all=False, show_infotable=False, show_anom_scores=False, show_preds=False, show_roc=False):
        """
        Evaluate the models seperately.

        This is done by functionality in evaluation.py
        """

        # calculate the anomaly scores
        self._calculate_anomaly_scores()

        # plot general information
        general_plotter = MiscPlotter(trainer=self)
        general_plotter.plot_losses(ylim=c.PLOT_YLIM_LOSSES)
        if show_infotable or show_all:
            general_plotter.plot_infotable()
        if show_anom_scores or show_all:
            general_plotter.plot_anomaly_scores()

        # plot the ROC curve and calculate optimal thresholds by checking many possible
        mses = [self.mse_lstm, self.mse_fft]
        thresholds = [c.THRESHOLD_LSTM, c.THRESHOLD_FFT]

        print("\nCalculating AUC...")
        roc_plotter = RocPlotter()
        for i in range(2):
            threshold_factors = np.arange(2, 0, -0.005)
            fps, tps, auc, f1_max = self._calculate_auc(mse=mses[i],
                                                        threshold=thresholds[i],
                                                        threshold_factors=threshold_factors)
            threshold = round(threshold_factors[f1_max[1]], 6)
            thresholds[i] = threshold * thresholds[i]
            roc_plotter.plot_single_roc(fps=fps, tps=tps, auc=auc, f1_max=f1_max)
        print(f"Thresholds: {thresholds}")
        if show_roc or show_all:
            roc_plotter.show()

        # get all time-periods where the anomaly score is above the threshold
        both_anomaly_times = []
        for i in range(2):
            anomaly_indexes = np.where(mses[i] > thresholds[i])[0]
            anomaly_sample_indexes = np.unique(anomaly_indexes // self.split_size)
            anomaly_times = find_timetuples_from_indexes(indexes=anomaly_sample_indexes,
                                                         max_index=self.data_3d.shape[0])
            both_anomaly_times.append(anomaly_times)

        if show_preds or show_all:
            PredictionsPlotter(file_path=f"{self.data_path}_{self.split_size}.csv",
                               sub_experiment_index=self.sub_experiment_index,
                               ylim=c.PLOT_YLIM_VIBRATION,
                               features=len(self.data_columns),
                               anomalies_real=anomalies[self.dataset_name][self.experiment_name],
                               anomalies_pred_lstm=both_anomaly_times[0],
                               anomalies_pred_fft=both_anomaly_times[1]
                               ).plot_experiment()

    def _calculate_anomaly_scores(self):
        self.data_pred_2d = self.model_lstm.predict(self.data_3d).reshape((-1, self.data_3d.shape[2]))
        self.fft_pred_2d = self.model_fft.predict(self.fft_3d).reshape((-1, self.fft_3d.shape[2]))
        self.mse_lstm = ((self.data_2d - self.data_pred_2d) ** 2).mean(axis=1)
        self.mse_fft = ((self.fft_2d - self.fft_pred_2d) ** 2)

    def _calculate_auc(self, mse, threshold, threshold_factors):

        all_anomaly_indexes = []
        fps = []
        tps = []
        f1_max = (0, 0)
        for threshold_factor in threshold_factors:
            anomaly_indexes = np.where(mse > threshold * threshold_factor)[0]
            anomaly_sample_indexes = np.unique(anomaly_indexes // self.split_size)
            all_anomaly_indexes.append(anomaly_sample_indexes)

        for i, indexes in enumerate(all_anomaly_indexes):
            tp, fp, fn, tn = get_tp_fp_fn_tn_from_indexes(pred_indexes=indexes,
                                                          max_index=self.data_3d.shape[0],
                                                          labels=self.labels)
            precision, recall, f1 = calculate_precision_recall_f1(tp=tp, fp=fp, fn=fn, tn=tn)
            if f1 > f1_max[0]:
                f1_max = (f1, i)
            fps.append(fp)
            tps.append(tp)
        p = tp + fn
        n = fp + tn

        # divide all the tps and fps by p and n
        tps = np.array(tps) / p
        fps = np.array(fps) / n

        # plot the ROC curve from the coordinates
        auc = np.trapz(y=tps, x=fps)

        return fps, tps, auc, f1_max


def evaluate_model(train, model_path=None):
    """ Run the experiment """
    dataset_path, data_columns, config_model_path = c.CLIENT_1['DATASET_PATH'], c.CLIENT_1['DATASET_COLUMNS'], c.CLIENT_1[
        'MODEL_PATH']

    if model_path is None:
        model_path = config_model_path

    # initialize the trainer and train/infere the model
    trainer = Training(data_path=dataset_path, data_columns=data_columns)
    if train:
        trainer.train_models(epochs=c.EPOCHS)
        trainer.save_models(dir_path=model_path)
    else:
        trainer.load_models(dir_path=model_path)

    trainer.evaluate(show_preds=True, show_roc=True)


if __name__ == '__main__':

    trainer = Training(data_path=c.CLIENT_1['DATASET_PATH'], data_columns=c.CLIENT_1['DATASET_COLUMNS'])
    # trainer.tune_models()
    trainer.train_models(epochs=10)
    trainer.save_models(dir_path=c.CLIENT_1['MODEL_PATH'].replace('model/', 'model/central/'))
    # trainer.load_models(dir_path="model/bearing/sabtain_2")
    trainer.evaluate(show_preds=True, show_roc=True, show_infotable=True, show_anom_scores=True)
