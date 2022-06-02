#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns

sns.set(color_codes=True)
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
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


# # SET HYPERPARAMETERS AND EXTERNAL VALUES

# In[3]:

import yaml
config = yaml.safe_load(open('../configs/config_baseline.yaml'))

# data
DATASET_PATH = config['DATASET_PATH']
MODEL_PATH = config['MODEL_PATH']
DATASET_COLUMN = config['DATASET_COLUMN']
TRAIN_SPLIT = config['TRAIN_SPLIT']
VAL_SPLIT = config['VAL_SPLIT']
# training
EPOCHS = config['EPOCHS']
BATCH_SIZE = config['BATCH_SIZE']
LEARNING_RATE = config['LEARNING_RATE']


# In[4]:


def readData(dataDir, setName, set1=False):
    files = os.listdir(dataDir)

    bearing1 = list()
    bearing2 = list()
    bearing3 = list()
    bearing4 = list()

    bearing_list = {setName + "_bearing-1": bearing1, setName + "_bearing-2": bearing2,
                    setName + "_bearing-3": bearing3, setName + "_bearing-4": bearing4}

    cols = ['bearing-1', 'bearing-2', 'bearing-3', 'bearing-4']

    for filename in files:

        dataset = pd.read_csv(os.path.join(dataDir, filename), sep='\t', header=None, names=cols)

        avg_df = dataset.groupby(dataset.index // 5).mean()

        for bearing in cols:
            dict_key = setName + "_" + bearing
            if dict_key in bearing_list:
                b = pd.DataFrame(avg_df[bearing]).T
                bearing_list[setName + "_" + bearing].append(b)

    return bearing_list


# In[5]:


data = readData(DATASET_PATH, "S2")

# # split data into training and test

# In[6]:


bearing1 = pd.concat(data[f"S2_bearing-{DATASET_COLUMN + 1}"])
split_len = int(len(bearing1) * TRAIN_SPLIT)
train = bearing1.iloc[0:split_len]
test = bearing1.iloc[split_len:]
train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

# In[7]:


# Standardarize train and test set
scaler = preprocessing.StandardScaler()
train_scaled = pd.DataFrame(scaler.fit_transform(train))
test_scaled = pd.DataFrame(scaler.transform(test))


# # prepare data for LSTM

# In[8]:


def temporalize(X, lookback):
    output_X = []
    for i in range(len(X) - lookback - 1):
        t = []
        for j in range(1, lookback + 1):
            # Gather past records upto the lookback period
            t.append(X[[(i + j + 1)], :])
        output_X.append(t)
    return output_X


def flatten(X):
    '''
    Flatten a 3D array.
    
    Input
    X            A 3D array for lstm, where the array is sample x timesteps x features.
    
    Output
    flattened_X  A 2D array, sample x features.
    '''
    flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
    for i in range(X.shape[0]):
        flattened_X[i] = X[i, (X.shape[1] - 1), :]
    return (flattened_X)


def scale(X, scaler):
    '''
    Scale 3D array.

    Inputs
    X            A 3D array for lstm, where the array is sample x timesteps x features.
    scaler       A scaler object, e.g., sklearn.preprocessing.StandardScaler, sklearn.preprocessing.normalize
    
    Output
    X            Scaled 3D array.
    '''
    for i in range(X.shape[0]):
        X[i, :, :] = scaler.transform(X[i, :, :])

    return X


# In[9]:


n_features = train_scaled.shape[1]
lookback = 20

# Temporalize the data
X = temporalize(X=train_scaled.values, lookback=lookback)
X_t = temporalize(X=test_scaled.values, lookback=lookback)

# In[10]:


X_train_lstm = np.array(X)
X_test_lstm = np.array(X_t)
X_train = X_train_lstm.reshape(X_train_lstm.shape[0], lookback, n_features)
X_test = X_test_lstm.reshape(X_test_lstm.shape[0], lookback, n_features)


def plot_and_evaluate(lstm_autoencoder):

    X_pred_train = lstm_autoencoder.predict(np.array(X_train))
    X_pred_test = lstm_autoencoder.predict(np.array(X_test))
    # concat both predictions
    x_pred = np.concatenate((X_pred_train, X_pred_test), axis=0)
    print(x_pred.shape)
    X_pred_train = pd.DataFrame(flatten(X_pred_train))
    X_pred_test = pd.DataFrame(flatten(X_pred_test))
    x_pred = pd.DataFrame(flatten(x_pred))

    # concatenate X_train and X_test
    # X_full = np.concatenate(flatten(X_train), flatten(X_test))

    scored_test = pd.DataFrame()
    mse_train = np.mean(np.abs(X_pred_train - flatten(X_train)), axis=1)
    mse_test = np.mean(np.abs(X_pred_test - flatten(X_test)), axis=1)
    # concate mse_train and mse_test
    mse = np.concatenate((mse_train, mse_test), axis=0)
    scored_test['RE'] = mse  # corrected error
    scored_test['Threshold'] = 1.0
    scored_test['Anomaly'] = scored_test['RE'] > scored_test['Threshold']  # corrected error

    # In[21]:

    scored_test.plot(figsize=(16, 8), color=['blue', 'red'], title="Failure of Bearing 1 (Set-2)")
    plt.show()

    return mse