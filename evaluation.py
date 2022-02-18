import os
# import winsound

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from config import *


def plot_all(res, loss, val_loss):
    res_train = res.iloc[:-2999, :]
    res_test = res.iloc[-3000:, :]
    res_chunk = res.iloc[5000:5000+3*SPLIT, :]

    # plot different graphics
    plt.figure(figsize=(12, 10))
    font_d = {'size': 12}

    plt.subplot(2, 2, 1)
    plt.title("Vibration and Prediction", fontdict=font_d)
    plt.plot(res['Data_1'])
    plt.plot(res['Pred_1'])
    plt.ylim(0, 3)

    plt.subplot(2, 2, 2)
    plt.title("Anomaly Score", fontdict=font_d)
    plt.plot(res['Loss_mae'])
    # plt.plot(res['Loss_avg'])
    plt.ylim(0, 1)
    plt.axhline(y=THRESHOLD, color='r', linestyle='-', label="Threshold")

    plt.subplot(2, 2, 3)
    plt.title("Vibration - Chunk", fontdict=font_d)
    plt.plot(res_chunk['Data_1'].tolist(), label="Data")
    plt.plot(res_chunk['Pred_1'].tolist(), label="Prediction")
    plt.legend()

    plt.subplot(2,2,4)
    plt.axis('off')
    text = f"Loss:       {loss:.3}\nVal_Loss: {val_loss:.3}\n\nEpochs: {EPOCHS}\nBatch: {BATCH_SIZE}\nLearn_r: {LEARNING_RATE:.0e}\nLR_Red: {LR_DECAY}\n\nLSTM: {2**(LAYERS_EXPONENT+2)}\nSplit: {SPLIT}\nLoss_fn: {LOSS}"
    plt.text(0.2, 0.9, text, ha='left', va='top', fontdict={'fontsize': 20})

    plt.savefig(f"results/{val_loss:.3e}_{SPLIT}_{2**LAYERS_EXPONENT}_{EPOCHS}_{BATCH_SIZE}_{LEARNING_RATE}.png")
    if os.path.exists('results/latest.png'):
        os.remove('results/latest.png')
    plt.savefig('results/latest.png')
    plt.show()
    return


def plot_sequence(sequence, title="Sequence", ylims=None, treshold=None, label=None):
    plt.plot(sequence, label=label)
    plt.title(f"{title}")
    plt.xlim(0, len(sequence))
    if ylims:
        plt.ylim(ylims[0], ylims[1])
    if treshold:
        plt.axhline(y=treshold, color='r', linestyle='-', label="Threshold")
    if label:
        plt.legend()
    plt.show()


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
    split_data(split_size=1000, path='data/bearing_dataset/bearings_3')