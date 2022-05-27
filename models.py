import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras import optimizers
from keras.layers import Input, Dense, Flatten, Reshape, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras.regularizers import l2

from util.config import c, c_client


def lstm_autoencoder_model(hp=None):
    """
    A function to create a specific autoencoder model. The model is a LSTM-autoencoder with a single hidden layer.

    The size of the hidden layer is determined by the number of features in the input.
    The size of the LSTM layers is dependant on the LAYERS_EXPONENT parameter in the files.

    :param X: the input data, which is used to configure the model correctly for the input size
    :return: the initialized model
    """

    # hyperparameter
    hp_learning_rate = hp.Choice('learning_rate', [1e-3, 3e-4, 1e-4, 3e-5, 1e-5]) if hp else 1e-4
    hp_hidden_size = hp.Int('hidden_size', 2, 12, step=2) if hp else 6

    print("HYPER-NAMES: type, learning_rate, hidden_size,")
    print(f"HYPERPARAMETERS:lstm;{hp_learning_rate};{hp_hidden_size}")

    # optimizer
    optimizer = optimizers.Adam(learning_rate=hp_learning_rate, clipnorm=1.0, clipvalue=0.5)

    # create the input layer
    timesteps = c.SPLIT
    num_features = len(c_client['DATASET_COLUMNS'])
    inputs = Input(shape=(timesteps, num_features))
    x = inputs

    # create the encoder LSTM layers
    for i in range(len(c.LAYER_SIZES) - 1):
        x = LSTM(c.LAYER_SIZES[i], activation='relu', return_sequences=True,
                 kernel_regularizer=l2(1e-7), activity_regularizer=l2(1e-7))(x)

    # create the last encoder layer with hp_hidden_size
    x = LSTM(hp_hidden_size, activation='relu', return_sequences=False,
             kernel_regularizer=l2(1e-7), activity_regularizer=l2(1e-7))(x)

    # pass the encoded data through a dense layer to the decoder
    x = RepeatVector(timesteps)(x)

    # create the first decoder layer with hp_hidden_size
    x = LSTM(hp_hidden_size, activation='relu', return_sequences=True,
             kernel_regularizer=l2(1e-7), activity_regularizer=l2(1e-7))(x)

    # create the decoder LSTM layers
    for i in reversed(range(len(c.LAYER_SIZES) - 1)):
        x = LSTM(c.LAYER_SIZES[i], activation='relu', return_sequences=True,
                 kernel_regularizer=l2(1e-7), activity_regularizer=l2(1e-7))(x)

    # create the plots layer
    output = TimeDistributed(Dense(num_features))(x)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=optimizer, loss='mse')

    return model


def fft_autoencoder_model(hp=None):
    """
    A function to create a specific autoencoder model for the FFT data. The model is a Denselayer-autoencoder.

    :param X: the input data, which is used to configure the model correctly for the input size
    :return: the initialized model
    """

    # hyperparameter
    hp_learning_rate = hp.Choice('learning_rate', [1e-3, 3e-4, 1e-4, 3e-5, 1e-5]) if hp else 1e-4
    hp_hidden_size = 6  # hp.Int('hidden_size', 1, 16, step=1) if hp else 6

    print("HYPER-NAMES: type, learning_rate, hidden_size,")
    print(f"HYPERPARAMETERS:fft;{hp_learning_rate};{hp_hidden_size}")

    # optimizer
    optimizer = optimizers.Adam(learning_rate=hp_learning_rate, clipnorm=1.0, clipvalue=0.5)

    # create the input layer
    timesteps = c.SPLIT
    num_features = len(c_client['DATASET_COLUMNS'])
    inputs = Input(shape=(timesteps, num_features))
    x = Flatten()(inputs)

    # create the encoder layers
    x = Dense(64, activation='relu', kernel_regularizer=l2(1e-7), activity_regularizer=l2(1e-7))(x)
    x = Dense(16, activation='relu', kernel_regularizer=l2(1e-7), activity_regularizer=l2(1e-7))(x)

    # pass the encoded data through a dense layer to the decoder
    x = Dense(hp_hidden_size, activation='relu', kernel_regularizer=l2(1e-7), activity_regularizer=l2(1e-7))(x)

    # create the decoder layers
    x = Dense(16, activation='relu', kernel_regularizer=l2(1e-7), activity_regularizer=l2(1e-7))(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(1e-7), activity_regularizer=l2(1e-7))(x)

    # create the plots layer
    x = Dense(timesteps * num_features)(x)
    output = Reshape((timesteps, num_features))(x)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=optimizer, loss='mse')

    return model