# import winsound

import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt

from helpers.config import c


def plot_all(results, loss, val_loss):
    """
    Plot the predictions and the actual values of the dataset. Also plot all relevant metrics such as loss and val_loss.

    :param results: dataframe with the predictions and metrics
    :param loss: loss of the trained model that is being evaluated
    :param val_loss: validation loss of the model that is being evaluated
    :return:
    """

    # take a chunk of the size of three measurements and plot the predictions and the actual values
    results_chunk = results.iloc[50 * c.SPLIT:53 * c.SPLIT, :]

    # plot different graphics
    plt.figure(figsize=(12, 10))
    font_d = {'size': 12}

    # create a subplot for the predictions and the actual values
    plt.subplot(2, 2, 1)
    plt.plot(results_chunk['Pred_1'], label='pred')
    plt.plot(results_chunk['Data_1'], label='actual')
    plt.legend(loc='upper left')
    plt.title('Predictions and Actual Values', fontdict=font_d)

    # create a subplot that plots the anomaly score and a threshold
    plt.subplot(2, 2, 2)
    plt.title("Anomaly Score", fontdict=font_d)
    plt.plot(results['Loss_mae'])
    # plt.plot(res['Loss_avg'])
    plt.ylim(0, 1)
    plt.axhline(y=c.THRESHOLD, color='r', linestyle='-', label="Threshold")

    # create a subplot for the predictions and the actual values of the chunk
    plt.subplot(2, 2, 3)
    plt.title("Vibration - Chunk", fontdict=font_d)
    plt.plot(results_chunk['Data_1'].tolist(), label="Data")
    plt.plot(results_chunk['Pred_1'].tolist(), label="Prediction")
    plt.legend()

    # create a subplot that includes all metrics in text form
    plt.subplot(2,2,4)
    plt.axis('off')
    text = f"Loss:       {loss[-1]:.3}\nVal_Loss: {val_loss[-1]:.3}\n\nEpochs: {c.EPOCHS}\nBatch: {c.BATCH_SIZE}\nLearn_r: {c.LEARNING_RATE:.0e}\nLR_Red: {c.LR_DECAY}\n\nLSTM: {2**(c.LAYERS_EXPONENT+2)}\nSplit: {c.SPLIT}"
    plt.text(0.2, 0.9, text, ha='left', va='top', fontdict={'fontsize': 20})

    plt.savefig(f"results/{val_loss[-1]:.3e}_{c.SPLIT}_{2**c.LAYERS_EXPONENT}_{c.EPOCHS}_{c.BATCH_SIZE}_{c.LEARNING_RATE:e.1}.png")
    if os.path.exists('results/latest.png'):
        os.remove('results/latest.png')
    plt.savefig('results/latest.png')
    plt.show()
    return


def split_data(split_size, path):
    chunk_list = []
    for file in os.scandir(path):
        df = pd.read_csv(f"{path}/{file.name}", sep='\t', header=None)
        # df.to_csv(f"data/bearing_dataset/absolutes_3_{split_size}.csv", mode='a', index=False, header=False)
        dfs = np.array_split(df, split_size)
        for df_chunk in dfs:
            mean_abs = np.array(df_chunk.abs().mean())
            mean_abs = pd.DataFrame(mean_abs.reshape(1, 4))
            mean_abs.to_csv(f"{path}_{split_size}.csv", mode='a', index=False, header=False)
        print('x', end='')
    # winsound.Beep(400,800)


if __name__ == '__main__':
    split_data(split_size=100, path='data/bearing_dataset/bearings_3')
