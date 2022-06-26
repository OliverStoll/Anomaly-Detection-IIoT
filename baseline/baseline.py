import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from datetime import datetime
import yaml
from numpy.random import seed
# from tensorflow import set_random_seed
from tensorflow.keras import optimizers
from keras import Sequential
from keras.models import Model
# from keras.utils import plot_model
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score

# In[2]:


# first = 'data/1st_test'
# second = 'data/2nd_test'
# third = 'data/4th_test/txt'


##########   SET HYPERPARAMETERS AND EXTERNAL VALUES


start = datetime.now()


config = yaml.safe_load(open('config_baseline.yaml'))

# data
DATASET_PATH = os.getenv('DATASET_PATH')
print(DATASET_PATH)
DATASET_COLUMN = config['DATASET_COLUMN']
EXPERIMENT_NAME = config['EXPERIMENT_NAME']
TRAIN_SPLIT = config['TRAIN_SPLIT']
VAL_SPLIT = config['VAL_SPLIT']
DOWNSAMPLE = config['DOWNSAMPLE']
# training
EPOCHS = config['EPOCHS']
BATCH_SIZE = config['BATCH_SIZE']
LEARNING_RATE = config['LEARNING_RATE']


#################    LOAD DATA    #################


def flatten(X):
    return X.reshape(X.shape[0], -1)


# get raw data
try:
    data = pd.read_csv(DATASET_PATH, usecols=DATASET_COLUMN)
except:
    print('Error: could not load data from Dataset_Path. Using "data" path instead.')
    data = pd.read_csv("./data.csv", usecols=DATASET_COLUMN)
# convert to numpy array
data = np.array(data)
data = data.reshape(-1, 20480)
print(data.shape)

# split data into training and test
split_len = int(len(data) * TRAIN_SPLIT)
print(split_len)
train = data[0:split_len]
test = data[split_len:]

# Standardarize train and test set
scaler = preprocessing.StandardScaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

# Get time
mid = datetime.now()
time_until_training = (mid-start).total_seconds()
print(f"TIME_UNTIL_TRAINING:{time_until_training:.2f}")

# reshape to 3d format for lstm     TODO: multiple features
num_features = len(DATASET_COLUMN)
X_train = train_scaled.reshape(train_scaled.shape[0], train_scaled.shape[1], num_features)
X_test = test_scaled.reshape(test_scaled.shape[0], test_scaled.shape[1], num_features)
print(X_train.shape, X_test.shape)

# # LSTM AE Anomaly Detector
lstm_autoencoder = Sequential()
lstm_autoencoder.add(LSTM(600, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
lstm_autoencoder.add(LSTM(250, activation='relu', return_sequences=False))
lstm_autoencoder.add(RepeatVector(X_train.shape[1]))  # Coding
lstm_autoencoder.add(LSTM(250, activation='relu', return_sequences=True))
lstm_autoencoder.add(LSTM(600, activation='tanh', return_sequences=True))
lstm_autoencoder.add(TimeDistributed(Dense(X_train.shape[2])))

lstm_autoencoder.summary()


adam = optimizers.Adam(LEARNING_RATE)
lstm_autoencoder.compile(loss='mse', optimizer=adam)

lstm_autoencoder_history = lstm_autoencoder.fit(X_train, X_train,
                                                epochs=EPOCHS,
                                                batch_size=BATCH_SIZE,
                                                validation_split=VAL_SPLIT,
                                                verbose=2).history


end = datetime.now()
print(f"TRAINING-DONE:{datetime.now()} - Time: {end-mid}")

# EVALUATION
X_pred_train = lstm_autoencoder.predict(np.array(X_train))
X_pred_train = pd.DataFrame(flatten(X_pred_train))

scored = pd.DataFrame(index=X_pred_train.index)
scored['RE'] = np.mean(np.abs(X_pred_train - flatten(X_train)), axis=1)

scored.describe()

X_pred_test = lstm_autoencoder.predict(np.array(X_test))
X_pred_test = pd.DataFrame(flatten(X_pred_test))

scored_test = pd.DataFrame(index=X_pred_test.index)
scored_test['RE'] = np.mean(np.abs(X_pred_test - flatten(X_test)), axis=1)  # corrected error
scored_test['Threshold'] = 1.0
scored_test['Anomaly'] = scored_test['RE'] > scored_test['Threshold']  # corrected error


# # MY EVALUATION
if '..' in DATASET_PATH:  # hacky way for manual testing
    lstm_autoencoder.save(f'../model/{EXPERIMENT_NAME}/baseline/lstm.h5')
scored_test['RE'].to_csv('baseline_anomaly_scores.csv', index=False)

# write string to file
result_times = f"RESULT TIMES: {EXPERIMENT_NAME},{(mid-start).total_seconds():.2f},{(end-mid).total_seconds():.2f}"
print(result_times)

with open('baseline_times.txt', 'w') as f:
    f.write(result_times + '\n')
