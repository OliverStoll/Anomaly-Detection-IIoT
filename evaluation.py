import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from functionality.config import c


def plot_debug(results, loss, val_loss, num_features):
    """
    Plot the predictions and the actual values of the dataset. Also plot all relevant metrics such as loss and val_loss.

    :param results: dataframe with the predictions and metrics
    :param loss: list of losses during training
    :param val_loss: list of validation-losses during training
    :param num_features: number of features in the dataset
    :return:
    """

    # take a chunk of the size of three measurements and plot the predictions and the actual values
    results_chunk = results.iloc[50 * c.SPLIT:53 * c.SPLIT, :]

    # plot different graphics
    plt.figure(figsize=(12, 10))
    font_d = {'size': 12}

    # create a subplot for the predictions and the actual values
    plt.subplot(2, 2, 1)
    for i in range(1):
        plt.plot(results[f'Pred_{i}'], label=f'pred_{i}', color=(1-i*0.1, 0, 0))
        plt.plot(results[f'Data_{i}'], label=f'actual_{i}', color=(0, 0, 1-i*0.1))
    plt.legend(loc='upper left')
    plt.title('Predictions and Actual Values', fontdict=font_d)

    # create a subplot that plots the anomaly score and a threshold
    plt.subplot(2, 2, 2)
    plt.title("Anomaly Score", fontdict=font_d)
    plt.plot(results['Loss_MSE'], label='Loss_MSE')
    plt.ylim(0, min(max(results['Loss_MSE']) * 1.2, 5 * c.THRESHOLD))
    plt.axhline(y=c.THRESHOLD, color='r', linestyle='-', label="Threshold")

    # create a subplot for the predictions and the actual values of the chunk
    plt.subplot(2, 2, 3)
    plt.title("Vibration - Chunk", fontdict=font_d)
    for i in range(num_features):
        plt.plot(results_chunk[f'Data_{i}'].tolist(), label=f"Data_{i}", color=(1-i*0.3, i*0.3, 0), linestyle='-')
        plt.plot(results_chunk[f'Pred_{i}'].tolist(), label=f"Prediction_{i}", color=(1-i*0.3, i*0.3, 0), linestyle='--')
    plt.legend()

    # create a subplot that includes all metrics in text form
    plt.subplot(2, 2, 4)
    plt.axis('off')
    text = f"Epochs: {c.EPOCHS}\nLoss:       {loss[-1]:.1e}\nVal_Loss: {val_loss[-1]:.1e}\n\n" \
           f"Batch: {c.BATCH_SIZE}\nLearn_r: {c.LEARNING_RATE:.0e}\nLR_Red: {c.LR_DECAY}" \
           f"\n\nLSTM: {c.LAYER_SIZES}\nSplit: {c.SPLIT}"
    plt.text(0.2, 0.9, text, ha='left', va='top', fontdict={'fontsize': 20})

    # create dir if it doesn't exist
    dir = f"archive/results/{c.CLIENT_1['DATASET_PATH']}"
    if not os.path.exists(dir):
        os.makedirs(dir)

    # save the figure
    plt.savefig(f"archive/results/{c.CLIENT_1['DATASET_PATH']}/{val_loss[-1]:.3e}_{c.SPLIT}.png")

    # show the figure
    plt.show()


def evaluate_model_lstm(model, data_3d, history):
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
    list_mse = ((data_2d - pred_2d) ** 2).mean(axis=1)
    results_df['Loss_MSE'] = list_mse

    # calculate the 1 percent quantile of the mse
    quantile = np.quantile(list_mse, 0.99)
    print("Quantile:", quantile)

    # determine the anomalies in the data based on the mse and the threshold
    results_df['Anomaly'] = results_df['Loss_MSE'] > c.THRESHOLD

    # plot the results
    plot_debug(results=results_df, loss=history['loss'], val_loss=history['val_loss'], num_features=data_2d.shape[1])


def evaluate_model_fft(model, fft_data_3d, plot_normalized=False):
    """
    Evaluate the fft-autoencoder model. Plot all relevant statistics in one image.

    :param model: the fft-autoencoder model to be evaluated.
    :return:
    """

    # get the predictions as 2d array from the model
    pred_2d = model.predict(fft_data_3d).reshape((-1, fft_data_3d.shape[2]))

    # reformat the data to a 2d array for evaluation
    data_2d = fft_data_3d.reshape((-1, fft_data_3d.shape[2]))

    # calculate the anomaly score and plot it
    mse = ((data_2d - pred_2d) ** 2)
    plt.plot(mse)
    plt.title("FFT Autoencoder Anomaly Score")
    plt.ylim(0, 5)
    plt.show()

    # normalize mse for multiple features efficiently to find one anomaly score
    scaler = MinMaxScaler()
    mse_s = scaler.fit_transform(mse)

    # if requested, plot the normalized mse
    if plot_normalized:
        plt.plot(mse_s)
        plt.title("FFT Autoencoder Anomaly Score (Normalized)")
        plt.show()



