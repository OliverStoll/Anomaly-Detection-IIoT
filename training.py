import tensorflow as tf
import pandas as pd
import numpy as np

from keras.layers import Input, Dense, Flatten, Reshape, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from tensorflow.keras import optimizers
from keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error

from evaluation import plot_all
from functionality.callbacks import scheduler
from functionality.config import c, client_c


def lstm_autoencoder_model(X):
    """
    A function to create a specific autoencoder model. The model is a LSTM-autoencoder with a single hidden layer.

    The size of the hidden layer is determined by the number of features in the input.
    The size of the LSTM layers is dependant on the LAYERS_EXPONENT parameter in the configs.

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
    print(f'CREATE FFT MODEL - INPUT: {X.shape} - LAYERS: {c.LAYER_SIZES}')

    # create the input layer
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    x = Flatten()(inputs)

    # create the encoder layers
    x = Dense(64, activation='relu', kernel_regularizer=l2(1e-7), activity_regularizer=l2(1e-7))(x)
    x = Dense(16, activation='relu', kernel_regularizer=l2(1e-7), activity_regularizer=l2(1e-7))(x)

    # pass the encoded data through a dense layer to the decoder
    x = Dense(8, activation='relu', kernel_regularizer=l2(1e-7), activity_regularizer=l2(1e-7))(x)

    # create the decoder layers
    x = Dense(16, activation='relu', kernel_regularizer=l2(1e-7), activity_regularizer=l2(1e-7))(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(1e-7), activity_regularizer=l2(1e-7))(x)

    # create the output layer
    x = Dense(X.shape[1] * X.shape[2])(x)
    output = Reshape((X.shape[1], X.shape[2]))(x)

    model = Model(inputs=inputs, outputs=output)
    return model


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
    train_data_3d = train_data.reshape((-1, c.SPLIT, train_data.shape[1]))
    data_3d = data.reshape((-1, c.SPLIT, data.shape[1]))

    return data, data_3d, train_data_3d


def calculate_fft(data_3d):
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


def init_models(train_data_3d):
    """
    Initialize the model with the training data. The training data needs to be formatted as a 3D array.

    The training data is split into training and validation data.

    :param train_data_3d: the training data, formatted as a 3D array (batch, timesteps, features)
    :return: the initialized model
    """

    # get both models from their respective functions
    fft = calculate_fft(train_data_3d)
    model = lstm_autoencoder_model(train_data_3d)
    model_fft = fft_autoencoder_model(fft)

    # create two adam optimizers for the models (could be one but not sure if safe to do so)
    opt = optimizers.Adam(learning_rate=c.LEARNING_RATE, clipnorm=1.0, clipvalue=0.5)
    opt_fft = optimizers.Adam(learning_rate=c.LEARNING_RATE, clipnorm=1.0, clipvalue=0.5)

    # compile the models
    model.compile(optimizer=opt, loss=c.LOSS_FN)
    model_fft.compile(optimizer=opt_fft, loss='mse')

    return model, model_fft


def train_models(model, model_fft, data_train_3d, fft_train_3d, epochs=1):
    """
    Train the model for a given number of epochs. The training data needs to be formatted as a 3D array.

    :param model: the model to train
    :param data_train_3d: the training data
    :param fft_train_3d: the training data for the FFT-layers
    :param epochs: the number of epochs to train for
    :return: the trained model
    """

    # initialize the learning rate scheduler
    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    # train the models
    _history_fft = model_fft.fit(fft_train_3d, fft_train_3d, epochs=epochs * 3, batch_size=c.BATCH_SIZE, validation_split=c.VAL_SPLIT).history
    _history = model.fit(data_train_3d, data_train_3d, epochs=epochs, batch_size=c.BATCH_SIZE, callbacks=[callback], validation_split=c.VAL_SPLIT).history

    return model, _history, _history_fft


def evaluate_model(model, data_3d, history):
    """
    Evaluate the model on the test data. This includes plotting all relevant metrics.

    :param model: The autoencoder model to evaluate
    :param data_3d: The test data to evaluate the model on, formatted as a 3D array
    :param history: The training history of the model
    :return:
    """

    # get the predictions as 2d array from the model
    pred_2d = model.predict(data_3d).reshape((-1, data_3d.shape[2]))

    # reformat the data to a 2D array for evaluation
    data_2d = data_3d.reshape((-1, data_3d.shape[2]))

    # store both predictions and data in a dataframe
    results_df = pd.DataFrame()
    for num_feature in range(data_2d.shape[1]):
        results_df[f'Data_{num_feature}'] = data_2d[:, num_feature]
        results_df[f'Pred_{num_feature}'] = pred_2d[:, num_feature]

    # calculate the mean squared error over all features
    results_df['Loss_MSE'] = ((data_2d - pred_2d) ** 2).mean(axis=1)
    results_df['mse'] = mean_squared_error(data_2d, pred_2d)

    # determine the anomalies in the data based on the mse and the threshold
    results_df['Anomaly'] = results_df['Loss_MSE'] > c.THRESHOLD

    # plot the results
    plot_all(results=results_df, loss=history['loss'], val_loss=history['val_loss'], num_features=data_2d.shape[1])


if __name__ == '__main__':
    data, data_3d, train_data_3d = load_and_normalize_data(data_path=f"data/{client_c['DATASET_PATH']}_{c.SPLIT}.csv",
                                                           columns=client_c['DATASET_COLUMNS'])
    model, model_fft = init_models(train_data_3d=train_data_3d)
    model, history, history_fft = train_models(model=model, model_fft=model_fft, data_train_3d=train_data_3d, fft_train_3d=calculate_fft(train_data_3d), epochs=c.EPOCHS)
    evaluate_model(model=model, data_3d=data_3d, history=history)
