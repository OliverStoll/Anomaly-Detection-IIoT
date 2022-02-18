import tensorflow as tf
import numpy as np
import pandas as pd
from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from tensorflow.keras import optimizers
from keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler

from config import *
from evaluation import *


def autoencoder_model(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    x = inputs
    x = LSTM(2**(LAYERS_EXPONENT+2), activation='relu', return_sequences=True,
             kernel_regularizer=l2(0.0000001), activity_regularizer=l2(0.0000001))(x)
    x = LSTM(2**LAYERS_EXPONENT, activation='relu', return_sequences=False,
             kernel_regularizer=l2(0.0000001), activity_regularizer=l2(0.0000001))(x)
    x = RepeatVector(X.shape[1])(x)
    x = LSTM(2**LAYERS_EXPONENT, activation='relu', return_sequences=True, kernel_regularizer=l2(0.0000001),
             activity_regularizer=l2(0.0000001))(x)
    x = LSTM(2**(LAYERS_EXPONENT+2), activation='relu', return_sequences=True, kernel_regularizer=l2(0.0000001),
             activity_regularizer=l2(0.0000001))(x)
    output = TimeDistributed(Dense(X.shape[2]))(x)
    model = Model(inputs=inputs, outputs=output)
    return model


def prepare_data(path='data/bearing_dataset'):
    names = ['1', '2', '3', '4'] if DATASET != 1 else ['1', '1_1', '2', '2_1', '3', '3_1', '4', '4_1']
    df = pd.read_csv(f'{path}/bearings_{DATASET}_{SPLIT}.csv', names=names)
    normal_len = int(len(df)*0.8)+(SPLIT-int(len(df)*0.8)%SPLIT)  # normal_len = len(df)
    normal_dfs = [df['2'][:normal_len], df['3'][:normal_len], df['4'][:normal_len]]
    all_dfs = [df['1'], df['2'], df['3'], df['4']]
    # split in training data and full data
    df_train = pd.concat(normal_dfs, ignore_index=True).tolist()  # training -> anomaly not included
    df = pd.concat(all_dfs, ignore_index=True).tolist()
    df_train = np.reshape(df_train, (len(df_train), 1))
    df = np.reshape(df, (len(df), 1))                       # reshape if only one feature
    # scale and reshape inputs for LSTM (samples, timesteps, features)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df)
    data_train = scaler.transform(df_train)
    data_3d = data.reshape(int(data.shape[0] / SPLIT), SPLIT, 1)
    data_train_3d = data_train.reshape(int(data_train.shape[0] / SPLIT), SPLIT, 1)

    return data, data_3d, data_train_3d


def init_model(data_train_3d):
    model = autoencoder_model(data_train_3d)
    opt = optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0, clipvalue=0.5)
    model.compile(optimizer=opt, loss=LOSS)
    return model


def train_model(model, data_train_3d, epochs=1):
    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    history = model.fit(data_train_3d, data_train_3d, epochs=epochs, batch_size=BATCH_SIZE, callbacks=[callback],
                        validation_split=0.1).history
    return model, history


def evaluate_model(model, data_3d, history):
    pred_seq = model.predict(data_3d).reshape(-1)
    data_seq = data_3d.reshape(-1)
    res = pd.DataFrame()
    res['Data_1'] = data_seq
    res['Pred_1'] = pred_seq
    res['Loss_mae'] = np.abs(pred_seq - data_seq)
    res['Threshold'] = THRESHOLD
    res['Anomaly'] = res['Loss_mae'] > res['Threshold']
    plot_all(res, loss=history[-1]['loss'][0], val_loss=history[-1]['val_loss'][0])
    # model.save(f"results/models/{history['val_loss'][-1]:.3e}_{SPLIT}_{2 ** LAYERS_EXPONENT}_{EPOCHS}_{BATCH_SIZE}_{LEARNING_RATE:.0e}.h5")
    # winsound.Beep(440, 800)


if __name__ == '__main__':
    data, data_3d, data_train_3d = prepare_data(path='data/bearing_dataset')
    model = init_model(data_train_3d=data_train_3d)
    model, history = train_model(model=model, data_train_3d=data_train_3d, epochs=1)
    evaluate_model(model=model, data_3d=data_3d, history=history)