import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.layers import Input, Dense, Flatten, Reshape, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from tensorflow.keras import optimizers
from keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from evaluation import *  # evaluate_model_lstm, evaluate_model_fft, find_timestretches_from_indexes
from util.callbacks import scheduler
from util.config import c, c_client


def calculate_fft_from_data(data_3d):
    """
    Calculate the FFT of the data. The Data needs to be a 3D array.

    :param data_3d: the 3d array of the data
    :return: the 3d array with the fft of the data
    """

    # calculate the FFT by iterating over the features and the samples
    fft = np.ndarray((data_3d.shape[0], data_3d.shape[1], data_3d.shape[2]))
    for j in range(data_3d.shape[2]):  # iterate over each feature
        for i in range(data_3d.shape[0]):  # iterate over each sample
            fft_origin = data_3d[i, :, j]
            fft_single = abs(np.fft.fft(fft_origin, axis=0))

            # append the single fft computation to the fft array
            fft[i, :, j] = fft_single

    return fft


def lstm_autoencoder_model(X):
    """
    A function to create a specific autoencoder model. The model is a LSTM-autoencoder with a single hidden layer.

    The size of the hidden layer is determined by the number of features in the input.
    The size of the LSTM layers is dependant on the LAYERS_EXPONENT parameter in the files.

    :param X: the input data, which is used to configure the model correctly for the input size
    :return: the initialized model
    """

    # print all relevant information
    # print(f'CREATE LSTM MODEL - LAYERS: {c.LAYER_SIZES}')

    # create the input layer
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    x = inputs

    # create the encoder LSTM layers
    for i in range(len(c.LAYER_SIZES) - 1):
        x = LSTM(c.LAYER_SIZES[i], activation='relu', return_sequences=True,
                 kernel_regularizer=l2(1e-7), activity_regularizer=l2(1e-7))(x)
    x = LSTM(c.LAYER_SIZES[-1], activation='relu', return_sequences=False,
             kernel_regularizer=l2(1e-7), activity_regularizer=l2(1e-7))(x)

    # pass the encoded data through a dense layer to the decoder
    x = RepeatVector(X.shape[1])(x)

    # create the decoder LSTM layers
    for i in reversed(range(len(c.LAYER_SIZES))):
        x = LSTM(c.LAYER_SIZES[i], activation='relu', return_sequences=True,
                 kernel_regularizer=l2(1e-7), activity_regularizer=l2(1e-7))(x)

    # create the output layer
    output = TimeDistributed(Dense(X.shape[2]))(x)
    model = Model(inputs=inputs, outputs=output)
    return model


def fft_autoencoder_model(X):
    """
    A function to create a specific autoencoder model for the FFT data. The model is a Denselayer-autoencoder.

    :param X: the input data, which is used to configure the model correctly for the input size
    :return: the initialized model
    """

    # print all relevant information
    # print(f'CREATE FFT MODEL - INPUT: {X.shape}')

    # create the input layer
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    x = Flatten()(inputs)

    # create the encoder layers
    x = Dense(64, activation='relu', kernel_regularizer=l2(1e-7), activity_regularizer=l2(1e-7))(x)
    x = Dense(16, activation='relu', kernel_regularizer=l2(1e-7), activity_regularizer=l2(1e-7))(x)

    # pass the encoded data through a dense layer to the decoder
    x = Dense(6, activation='relu', kernel_regularizer=l2(1e-7), activity_regularizer=l2(1e-7))(x)

    # create the decoder layers
    x = Dense(16, activation='relu', kernel_regularizer=l2(1e-7), activity_regularizer=l2(1e-7))(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(1e-7), activity_regularizer=l2(1e-7))(x)

    # create the output layer
    x = Dense(X.shape[1] * X.shape[2])(x)
    output = Reshape((X.shape[1], X.shape[2]))(x)

    model = Model(inputs=inputs, outputs=output)
    return model


class Training:
    def __init__(self, data_path, data_columns):
        self.data_path = data_path
        self.dataset_name = data_path.split('/')[-2]
        self.experiment_name = data_path.split('/')[-1]
        self.sub_experiment_index = int(data_columns[0] / len(data_columns))
        self.data_columns = data_columns
        self.split_size = c.SPLIT
        self.batch_size = c.BATCH_SIZE
        self.val_split = c.VAL_SPLIT
        self.train_split = c.TRAIN_SPLIT
        self.labels = anomalies[self.dataset_name][self.experiment_name]['bearing-0']
        self.data_3d, self.data_train_3d = self.load_and_normalize_data()
        self.data_2d = self.data_3d.reshape((-1, self.data_3d.shape[2]))
        self.fft_3d = calculate_fft_from_data(self.data_3d)
        self.fft_2d = self.fft_3d.reshape((-1, self.fft_3d.shape[2]))
        self.fft_train_3d = calculate_fft_from_data(self.data_train_3d)
        self.model_lstm, self.model_fft = self.initialize_models()
        self.history_lstm = {'loss': [], 'val_loss': []}
        self.history_fft = {'loss': [], 'val_loss': []}
        self.mse_lstm = []
        self.mse_fft = []
        self.data_pred_2d = []
        self.fft_pred_2d = []
        self.callbacks = [tf.keras.callbacks.LearningRateScheduler(scheduler)]  # initialize the learning rate scheduler

    def load_and_normalize_data(self):
        """
        Prepare the data for training and evaluation. All features need to be extracted and named.

        The data is normalized and possibly split into training and testing data 80/20.

        :return: the full data, the training data and the 3d array of the training data
        """

        # read the data from the csv file
        df = pd.read_csv(f"{self.data_path}_{self.split_size}.csv", usecols=self.data_columns)

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

    def initialize_models(self):
        """
        Initialize the model with the training data. The training data needs to be formatted as a 3D array.

        The training data is split into training and validation data.

        :return: the initialized model
        """

        # get both models from their respective functions
        model_lstm = lstm_autoencoder_model(self.data_train_3d)
        model_fft = fft_autoencoder_model(self.fft_train_3d)

        # create two adam optimizers for the models (could be one but not sure if safe to do so)
        opt = optimizers.Adam(learning_rate=c.LEARNING_RATE, clipnorm=1.0, clipvalue=0.5)
        opt_fft = optimizers.Adam(learning_rate=c.LEARNING_RATE, clipnorm=1.0, clipvalue=0.5)

        # compile the models
        model_lstm.compile(optimizer=opt, loss=c.LOSS_FN)
        model_fft.compile(optimizer=opt_fft, loss='mse')

        return model_lstm, model_fft

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

    def calculate_anomaly_scores(self):
        self.data_pred_2d = self.model_lstm.predict(self.data_3d).reshape((-1, self.data_3d.shape[2]))
        self.fft_pred_2d = self.model_fft.predict(self.fft_3d).reshape((-1, self.fft_3d.shape[2]))
        self.mse_lstm = ((self.data_2d - self.data_pred_2d) ** 2).mean(axis=1)
        self.mse_fft = ((self.fft_2d - self.fft_pred_2d) ** 2)

    def calculate_auc(self, threshold_factors):

        all_anomaly_indexes = []
        auc_coords = []
        fps = []
        tps = []
        for threshold_factor in threshold_factors:
            anomaly_indexes_lstm = np.where(self.mse_lstm > c.THRESHOLD_LSTM * threshold_factor)[0]
            anomaly_sample_indexes_lstm = np.unique(anomaly_indexes_lstm // self.split_size)
            all_anomaly_indexes.append(anomaly_sample_indexes_lstm)

        for indexes in all_anomaly_indexes:
            tp, fp, fn, tn = get_tp_fp_fn_tn_from_indexes(pred_indexes=indexes,
                                                          max_index=self.data_3d.shape[0],
                                                          labels=self.labels)
            fps.append(fp)
            tps.append(tp)
        p = tp + fn
        n = fp + tn

        # divide all the tps and fps by p and n
        tps = np.array(tps) / p
        fps = np.array(fps) / n

        # plot the ROC curve from the coordinates
        auc = np.trapz(y=tps, x=fps)

        return fps, tps, auc


    def evaluation(self, show_infotable=False, show_anomaly_scores=True, show_experiment=True):
        """
        Evaluate the models seperately.

        This is done by functionality in evaluation.py
        """

        # evaluate the models seperately
        if show_infotable:
            plot_infotable(trainer=self)

        if show_anomaly_scores:
            plot_anomaly_scores(mse_lstm=self.mse_lstm, mse_fft=self.mse_fft)

        # get all indexes where the anomaly score is above the threshold, and from that the corresponding sample indexes
        anomaly_indexes_lstm = np.where(self.mse_lstm > c.THRESHOLD_LSTM)[0]
        anomaly_sample_indexes_lstm = np.unique(anomaly_indexes_lstm // self.split_size)
        anomaly_times_lstm = find_timetuples_from_indexes(indexes=anomaly_sample_indexes_lstm,
                                                          max_index=self.data_3d.shape[0])

        anomaly_indexes_fft = np.where(self.mse_fft > c.THRESHOLD_FFT)[0]
        anomaly_sample_indexes_fft = np.unique(anomaly_indexes_fft // self.split_size)

        anomaly_times_fft = find_timetuples_from_indexes(indexes=anomaly_sample_indexes_fft,
                                                         max_index=self.data_3d.shape[0])

        plotter = ExperimentPlotter(file_path=f"{self.data_path}_{self.split_size}.csv",
                                    sub_experiment_index=self.sub_experiment_index,
                                    ylim=c.PLOT_YLIM,
                                    features=len(self.data_columns),
                                    anomalies_real=anomalies[self.dataset_name][self.experiment_name],
                                    anomalies_pred_lstm=anomaly_times_lstm,
                                    anomalies_pred_fft=anomaly_times_fft)

        if show_experiment:
            plotter.plot_experiment()

        tp, fp, fn, tn = get_tp_fp_fn_tn_from_indexes(pred_indexes=anomaly_sample_indexes_lstm,
                                                      max_index=self.data_3d.shape[0],
                                                      labels=self.labels)

        precision, recall, f1 = calculate_precision_recall_f1(tp=tp, fp=fp, fn=fn, tn=tn)
        print(f"Precision: {precision: .3f}")
        print(f"Recall: {recall: .3f}")
        print(f"F1: {f1: .3f}")

        fps, tps, auc = self.calculate_auc(threshold_factors=np.arange(2, 0, -0.005))
        print(f"AUC: {auc: .3f}")
        plot_roc(fps=fps, tps=tps, auc=auc)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # disable GPU usage

    # define the experiment for testing
    train = False
    experiment, data_columns = "bearing/experiment-2", [0]

    # initialize the trainer and train/infere the model
    trainer = Training(data_path=f"data/{experiment}", data_columns=data_columns)
    if train:
        trainer.train_models(epochs=c.EPOCHS)
        trainer.save_models(dir_path=f"models/{experiment}/{data_columns}")
    else:
        trainer.load_models(dir_path=f"models/{experiment}/{data_columns}")

    trainer.calculate_anomaly_scores()
    trainer.evaluation(show_anomaly_scores=train)
