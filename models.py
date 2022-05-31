import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras import optimizers
from keras.layers import Input, Dense, Flatten, Reshape, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras.regularizers import l2

from util.config import c, c_client


def lstm_autoencoder_model(hp=None):
    """
    A function to create a LSTM-autoencoder with a single hidden layer.

    The size of the hidden layer is determined by the number of features in the input.
    The size of the LSTM layers is dependant on the LAYERS_EXPONENT parameter in the files.

    :param X: the input data, which is used to configure the model correctly for the input size
    :return: the initialized model
    """

    # default hyperparameters
    learning_rate = c.LEARNING_RATE
    outer_layer_size = c.OUTER_LAYER_SIZE
    layers_amount = c.LAYER_AMOUNT
    hidden_size = c.HIDDEN_LAYER_SIZE

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
    timesteps = c.SPLIT
    num_features = len(c_client['DATASET_COLUMNS'])
    inputs = Input(shape=(timesteps, num_features))
    x = inputs

    # create the encoder LSTM layers
    for layer_size in layer_sizes[::-1]:
        x = LSTM(layer_size, activation='tanh', return_sequences=True,
                 kernel_regularizer=l2(1e-7), activity_regularizer=l2(1e-7))(x)

    # create the last encoder layer with hp_hidden_size
    x = LSTM(hidden_size, activation='tanh', return_sequences=False,
             kernel_regularizer=l2(1e-7), activity_regularizer=l2(1e-7))(x)

    # pass the encoded data through a dense layer to the decoder
    x = RepeatVector(timesteps)(x)

    # create the first decoder layer with hp_hidden_size
    x = LSTM(hidden_size, activation='tanh', return_sequences=True,
             kernel_regularizer=l2(1e-7), activity_regularizer=l2(1e-7))(x)

    # create the decoder LSTM layers
    for layer_size in layer_sizes:
        x = LSTM(layer_size, activation='tanh', return_sequences=True,
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

    # default hyperparameters
    learning_rate = c.LEARNING_RATE
    outer_layer_size = c.OUTER_LAYER_SIZE
    layers_amount = c.LAYER_AMOUNT
    hidden_size = 6

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
    timesteps = c.SPLIT
    num_features = len(c_client['DATASET_COLUMNS'])
    inputs = Input(shape=(timesteps, num_features))
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
    x = Dense(timesteps * num_features)(x)
    output = Reshape((timesteps, num_features))(x)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=optimizer, loss='mse')

    return model