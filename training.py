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

from evaluation import evaluate_model_lstm, evaluate_model_fft
from util.callbacks import scheduler
from util.config import c, client_c


def load_and_normalize_data(data_path, columns):
    """
    Prepare the data for training and evaluation. All features need to be extracted and named.

    The data is normalized and split into training and testing data 80/20.

    :param data_path: file path to the csv data file
    :param columns: the columns to extract from the csv file
    :return: the full data, the training data and the 3d array of the training data
    """

    # read the data from the csv file
    df = pd.read_csv(data_path, sep=';', usecols=columns, header=None)

    # drop the last rows that are not a full split anymore
    if len(df) % c.SPLIT != 0:
        df = df.iloc[:-(len(df) % c.SPLIT)]

    # split the data into training and testing data 80/20 (only for bearing)
    if 'bearing' in c.CLIENT_1['DATASET_PATH']:
        split_len = int(len(df) * 0.8) + (c.SPLIT - int(len(df) * 0.8) % c.SPLIT)
    else:
        split_len = len(df)

    # split the data frame into multiple lists
    data = df.iloc[:, :].values
    train_data = df[:split_len].iloc[:, :].values

    # normalize the data
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    train_data = scaler.transform(train_data)

    # reshape data to 3d arrays with the second value equal to the number of timesteps (SPLIT)
    data_3d = data.reshape((-1, c.SPLIT, data.shape[1]))
    train_data_3d = train_data.reshape((-1, c.SPLIT, train_data.shape[1]))

    return data_3d, train_data_3d


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
    print(f'CREATE MODEL - INPUT: {X.shape} - LAYERS: {c.LAYER_SIZES}')

    # create the input layer
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    x = inputs

    # create the encoder LSTM layers
    for i in range(len(c.LAYER_SIZES)-1):
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
    print(f'CREATE FFT MODEL - INPUT: {X.shape}')

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
        self.data_3d, self.data_train_3d = load_and_normalize_data(data_path, data_columns)
        self.fft_3d = calculate_fft_from_data(self.data_3d)
        self.fft_train_3d = calculate_fft_from_data(self.data_train_3d)
        self.model_lstm, self.model_fft = self.initialize_models()
        self.history_lstm = {'loss': [], 'val_loss': []}
        self.history_fft = {'loss': [], 'val_loss': []}
        self.callbacks = [tf.keras.callbacks.LearningRateScheduler(scheduler)]  # initialize the learning rate scheduler

    def initialize_models(self):
        """
        Initialize the model with the training data. The training data needs to be formatted as a 3D array.

        The training data is split into training and validation data.

        :param train_data_3d: the training data, formatted as a 3D array (batch, timesteps, features)
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
                                            batch_size=c.BATCH_SIZE,
                                            callbacks=self.callbacks,
                                            validation_split=c.VAL_SPLIT).history
        _history_fft = self.model_fft.fit(self.fft_train_3d,
                                          self.fft_train_3d,
                                          epochs=epochs,
                                          batch_size=c.BATCH_SIZE,
                                          validation_split=c.VAL_SPLIT).history

        self.history_lstm['loss'] += _history_lstm['loss']
        self.history_lstm['val_loss'] += _history_lstm['val_loss']
        self.history_fft['loss'] += _history_fft['loss']
        self.history_fft['val_loss'] += _history_fft['val_loss']

    def evaluation(self):
        """
        Debug Evaluation function.
        """
        evaluate_model_lstm(model=self.model_lstm, data_3d=self.data_3d, history=self.history_lstm)
        evaluate_model_fft(model=self.model_fft, fft_data_3d=self.fft_3d)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # disable GPU usage

    trainer = Training(data_path=f"data/{client_c['DATASET_PATH']}_{c.SPLIT}.csv",
                       data_columns=client_c['DATASET_COLUMNS'])
    trainer.train_models(epochs=c.EPOCHS)
    trainer.evaluation()
