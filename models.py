import os
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras import optimizers
from keras.layers import Input, Dense, Flatten, Reshape, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras.regularizers import l2

from util.config import c, client_config


def lstm_autoencoder_model(hp=None):
    """
    A function to create a LSTM-autoencoder.

    The size of the hidden layer is determined by the number of features in the input.
    The size of the LSTM layers is dependant on the LAYERS_EXPONENT parameter in the files.

    :param X: the input data, which is used to configure the model correctly for the input size
    :return: the initialized model
    """

    # default hyperparameters
    learning_rate = c.LSTM['LEARNING_RATE']
    outer_layer_size = c.LSTM['OUTER_LAYER_SIZE']
    layers_amount = c.LSTM['LAYER_AMOUNT']
    hidden_size = c.LSTM['HIDDEN_LAYER_SIZE']

    # hyperparameter tuning
    if hp:
        learning_rate = hp.Choice('learning_rate', [3e-2, 1e-2, 3e-2, 1e-3, 3e-4])
        outer_layer_size = hp.Choice('outer_layer_size', [32, 64, 128, 256, 512])
        layers_amount = hp.Choice('layers_amount', [1, 2, 3, 4])
        print(f"HYPERPARAMETERS:lstm;{learning_rate};{outer_layer_size};{layers_amount}")

    # calculate the layer sizes from hyperparameters
    layer_shrinking_factor = outer_layer_size / hidden_size
    layer_sizes = [int(hidden_size * layer_shrinking_factor ** ((i+1)/layers_amount)) for i in range(layers_amount)]

    # optimizer
    optimizer = optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0, clipvalue=0.5)

    # create the input layer
    timesteps = os.getenv('WINDOW_SIZE', c.WINDOW_SIZE)
    num_features = c.NUM_FEATURES
    inputs = Input(shape=(timesteps, num_features))
    x = inputs

    layer_kwdict ={'activation': 'relu',
                   'kernel_regularizer':l2(1e-7),
                   'activity_regularizer':l2(1e-7)}

    # create the encoder LSTM layers
    for layer_size in layer_sizes[::-1]:
        x = LSTM(layer_size, return_sequences=True, **layer_kwdict)(x)

    # create the last encoder layer with hp_hidden_size
    x = LSTM(hidden_size, return_sequences=False, **layer_kwdict)(x)
    x = RepeatVector(timesteps)(x)
    # x = Dense(hidden_size, **layer_kwdict)(x)  # use dense layer instead for hidden layer


    # create the decoder LSTM layers
    for layer_size in layer_sizes:
        x = LSTM(layer_size, return_sequences=True, **layer_kwdict)(x)

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

    # default hyperparameters
    learning_rate = c.FFT['LEARNING_RATE']
    outer_layer_size = c.FFT['OUTER_LAYER_SIZE']
    layers_amount = c.FFT['LAYER_AMOUNT']
    hidden_size = c.FFT['HIDDEN_LAYER_SIZE']

    # hyperparameter tuning
    if hp:
        learning_rate = hp.Choice('learning_rate', [3e-2, 1e-2, 3e-2, 1e-3, 3e-4])
        outer_layer_size = hp.Choice('outer_layer_size', [32, 64, 128, 256, 512])
        layers_amount = hp.Choice('layers_amount', [1, 2, 3, 4])
        print(f"HYPERPARAMETERS:fft;{learning_rate};{outer_layer_size};{layers_amount}")

    # calculate the layer sizes from hyperparameters
    layer_shrinking_factor = outer_layer_size / hidden_size
    layer_sizes = [int(hidden_size * layer_shrinking_factor ** ((i + 1) / layers_amount)) for i in range(layers_amount)]

    # optimizer
    optimizer = optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0, clipvalue=0.5)

    # create the input layer
    timesteps = os.getenv('WINDOW_SIZE', c.WINDOW_SIZE)
    inputs = Input(shape=(timesteps, c.NUM_FEATURES))
    x = Flatten()(inputs)

    # create the encoder layers
    for layer_size in layer_sizes[::-1]:
        x = Dense(layer_size, activation='relu', kernel_regularizer=l2(1e-7), activity_regularizer=l2(1e-7))(x)

    # pass the encoded data through a dense layer to the decoder
    x = Dense(hidden_size, activation='relu', kernel_regularizer=l2(1e-7), activity_regularizer=l2(1e-7))(x)

    # create the decoder layers
    for layer_size in layer_sizes:
        x = Dense(layer_size, activation='relu', kernel_regularizer=l2(1e-7), activity_regularizer=l2(1e-7))(x)

    # create the plots layer
    x = Dense(timesteps * c.NUM_FEATURES)(x)
    output = Reshape((timesteps, c.NUM_FEATURES))(x)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=optimizer, loss='mse')

    return model


if __name__ == '__main__':
    model = lstm_autoencoder_model()
    model.summary()