import tensorflow as tf
import pandas as pd
import numpy as np

from keras.layers import Input, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from tensorflow.keras import optimizers
from keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error

from evaluation import plot_all
from functionality.callbacks import scheduler
from functionality.config import c, client_c


def autoencoder_model(X):
    """
    A function to create a specific autoencoder model. The model is a LSTM-autoencoder with a single hidden layer.
    The size of the hidden layer is determined by the number of features in the input.
    The size of the LSTM layers is dependant on the LAYERS_EXPONENT parameter in the configs.

    :param X: the input data, which is used to configure the model correctly for the input size
    :return: the initialized model
    """
    # print all relevant information
    print(f'INPUT: {X.shape} - LAYERS: {c.LAYER_SIZES}')

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


def prepare_data(data_path, columns):
    """
    Prepare the data for training and evaluation. All features need to be extracted and named. The data is
    normalized and split into training and testing data 80/20.

    :param data_path: file path to the csv data file
    :param columns: the columns to extract from the csv file
    :return: the full data, the training data and the 3d array of the training data
    """
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
    # TODO: Standard or MinMax?
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    train_data = scaler.transform(train_data)

    # reshape data to 3d arrays with the second value equal to the number of timesteps (SPLIT)
    train_data_3d = train_data.reshape((-1, c.SPLIT, train_data.shape[1]))
    data_3d = data.reshape((-1, c.SPLIT, data.shape[1]))

    # TODO: calculate the FFT of the data
    # fft0 = abs(np.fft.fft(data_3d[0, :, 0]))
    # fft1 = abs(np.fft.fft(data[:, 1]))

    return data, data_3d, train_data_3d


def init_model(train_data_3d):
    """
    Initialize the model with the training data. The training data needs to be formatted as a 3D array.
    The training data is split into training and validation data.

    :param train_data_3d: the training data, formatted as a 3D array (batch, timesteps, features)
    :return: the initialized model
    """
    model = autoencoder_model(train_data_3d)
    opt = optimizers.Adam(learning_rate=c.LEARNING_RATE, clipnorm=1.0, clipvalue=0.5)
    model.compile(optimizer=opt, loss=c.LOSS_FN)
    return model


def train_model(model, data_train_3d, epochs=1):
    """
    Train the model for a given number of epochs. The training data needs to be formatted as a 3D array for
    the LSTM-layers.

    :param model: the model to train
    :param data_train_3d: the training data
    :param epochs: the number of epochs to train for
    :return: the trained model
    """
    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    _history = model.fit(data_train_3d, data_train_3d, epochs=epochs, batch_size=c.BATCH_SIZE,
                         callbacks=[callback], validation_split=c.VAL_SPLIT).history
    return model, _history


def evaluate_model(model, data_3d, history):
    """
    Evaluate the model on the test data. This includes plotting all relevant metrics.

    :param model: The autoencoder model to evaluate
    :param data_3d: The test data to evaluate the model on, formatted as a 3D array
    :param history: The training history of the model
    :return:
    """
    # TODO: generalize for multiple features
    # get the predictions from the model
    pred_2d = model.predict(data_3d).reshape((-1, data_3d.shape[2]))
    # reformat the data_3d to a 2D array for evaluation
    data_2d = data_3d.reshape((-1, data_3d.shape[2]))

    results_df = pd.DataFrame()
    for num_feature in range(data_2d.shape[1]):
        results_df[f'Data_{num_feature}'] = data_2d[:, num_feature]
        results_df[f'Pred_{num_feature}'] = pred_2d[:, num_feature]
    # calculate the mean squared error over all features
    results_df['Loss_MSE'] = ((data_2d - pred_2d) ** 2).mean(axis=1)
    results_df['mse'] = mean_squared_error(data_2d, pred_2d)
    results_df['Anomaly'] = results_df['Loss_MSE'] > c.THRESHOLD
    plot_all(results=results_df, loss=history['loss'], val_loss=history['val_loss'], num_features=data_2d.shape[1])
    # model.save(f"model/model/{history['val_loss'][-1]:.3e}_{SPLIT}_{2 ** LAYERS_EXPONENT}_{EPOCHS}_{BATCH_SIZE}_{LEARNING_RATE:.0e}.h5")
    # winsound.Beep(440, 800)


if __name__ == '__main__':
    data, data_3d, train_data_3d = prepare_data(data_path=f"data/{client_c['DATASET_PATH']}_{c.SPLIT}.csv",
                                                columns=client_c['DATASET_COLUMNS'])
    model = init_model(train_data_3d=train_data_3d)
    model, history = train_model(model=model, data_train_3d=train_data_3d, epochs=c.EPOCHS)
    evaluate_model(model=model, data_3d=data_3d, history=history)
