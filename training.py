import tensorflow as tf
import numpy as np
import pandas as pd
import yaml
from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from tensorflow.keras import optimizers
from keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler

from evaluation import *

# read configs from yaml file
config = yaml.safe_load(open("configs/config.yaml"))


def autoencoder_model(X):
    """
    A function to create a specific autoencoder model. The model is a LSTM-autoencoder with a single hidden layer.
    The size of the hidden layer is determined by the number of features in the input.
    The size of the LSTM layers is dependant on the LAYERS_EXPONENT parameter in the configs.

    :param X: the input data, which is used to configure the model correctly for the input size
    :return: the initialized model
    """
    # print all relevant information
    print(f'Input shape: {X.shape}')
    print(f'Layers size: {2 ** (LAYERS_EXPONENT+2)}, {2 ** LAYERS_EXPONENT}')

    # create the input layer
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    x = inputs

    # create the LSTM layers
    x = LSTM(2 ** (LAYERS_EXPONENT + 2), activation='relu', return_sequences=True,
             kernel_regularizer=l2(0.0000001), activity_regularizer=l2(0.0000001))(x)
    x = LSTM(2 ** LAYERS_EXPONENT, activation='relu', return_sequences=False,
             kernel_regularizer=l2(0.0000001), activity_regularizer=l2(0.0000001))(x)
    x = RepeatVector(X.shape[1])(x)
    x = LSTM(2 ** LAYERS_EXPONENT, activation='relu', return_sequences=True, kernel_regularizer=l2(0.0000001),
             activity_regularizer=l2(0.0000001))(x)
    x = LSTM(2 ** (LAYERS_EXPONENT + 2), activation='relu', return_sequences=True, kernel_regularizer=l2(0.0000001),
             activity_regularizer=l2(0.0000001))(x)

    # create the output layer
    output = TimeDistributed(Dense(X.shape[2]))(x)
    model = Model(inputs=inputs, outputs=output)
    return model


def prepare_data(columns, data_path=f'data/bearing_dataset/bearings_2_10.csv'):
    """
    Prepare the data for training and evaluation. All features need to be extracted and named. The data is
    normalized and split into training and testing data 80/20.

    :param data_path: file path to the csv data file
    :param columns: the columns to extract from the csv file
    :return: the full data, the training data and the 3d array of the training data
    """
    # TODO: include metadata features
    df = pd.read_csv(data_path, names=columns)
    split_len = int(len(df) * 0.8) + (SPLIT - int(len(df) * 0.8) % SPLIT)

    # extract the columns with the features by index using the columns list
    df = df.iloc[:, columns]

    # split the data frame into multiple lists
    data = df.iloc[:, :].values
    train_data = df[:split_len].iloc[:, :].values

    # normalize the data
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    train_data = scaler.transform(train_data)

    # reshape data to 3d arrays with the second value equal to the number of timesteps (SPLIT)
    train_data_3d = train_data.reshape((-1, SPLIT, train_data.shape[1]))
    data_3d = data.reshape((-1, SPLIT, data.shape[1]))

    return data, data_3d, train_data_3d


def init_model(data_train_3d):
    """
    Initialize the model with the training data. The training data needs to be formatted as a 3D array.
    The training data is split into training and validation data.

    :param data_train_3d: the training data
    :return: the initialized model
    """
    model = autoencoder_model(data_train_3d)
    opt = optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0, clipvalue=0.5)
    model.compile(optimizer=opt, loss=LOSS_FN)
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
    _history = model.fit(data_train_3d, data_train_3d, epochs=epochs, batch_size=BATCH_SIZE,
                         callbacks=[callback], validation_split=0.1).history
    return model, _history


def evaluate_model(model, data_3d, history, columns):
    """
    Evaluate the model on the test data. This includes plotting all relevant metrics.

    :param model: The autoencoder model to evaluate
    :param data_3d: The test data to evaluate the model on, formatted as a 3D array
    :param history: The training history of the model
    :return:
    """
    # TODO: generalize for multiple features
    # for each feature, predict the values using the model and plot the results
    for column in columns:
        # get the feature data
        feature_data = data_3d[:, :, columns.index(column)]
        # get the predicted data
        predicted_data = model.predict(feature_data)
        # get the real data
        real_data = feature_data[:, -1, :]
        # plot the results
        plot_all(real_data, predicted_data, column)

    pred_seq = model.predict(data_3d).reshape(-1)
    data_seq = data_3d.reshape(-1)
    res = pd.DataFrame()
    res['Data_1'] = data_seq
    res['Pred_1'] = pred_seq
    res['Loss_mae'] = np.abs(pred_seq - data_seq)
    res['Threshold'] = THRESHOLD
    res['Anomaly'] = res['Loss_mae'] > res['Threshold']
    plot_all(results=res, loss=history['loss'], val_loss=history['val_loss'])
    # model.save(f"results/models/{history['val_loss'][-1]:.3e}_{SPLIT}_{2 ** LAYERS_EXPONENT}_{EPOCHS}_{BATCH_SIZE}_{LEARNING_RATE:.0e}.h5")
    # winsound.Beep(440, 800)


if __name__ == '__main__':
    data, data_3d, train_data_3d = prepare_data(columns=DATASET_COLUMNS, data_path=DATASET_PATH)
    model = init_model(data_train_3d=train_data_3d)
    model, history = train_model(model=model, data_train_3d=train_data_3d, epochs=100)
    evaluate_model(model=model, data_3d=data_3d, history=history, columns=DATASET_COLUMNS)
