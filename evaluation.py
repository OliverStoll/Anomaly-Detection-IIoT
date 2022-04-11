# import winsound

import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt

from scripts.config import c


def plot_all(results, loss, val_loss, num_features):
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
    plt.ylim(0, min(max(results['Loss_MSE']) * 1.2, 1))
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

    plt.savefig(f"archive/results/{val_loss[-1]:.3e}_{c.SPLIT}_{c.LAYER_SIZES}_{c.EPOCHS}_"
                f"{c.BATCH_SIZE}_{c.LEARNING_RATE:.1e}.png")
    plt.show()
    return


def split_data_nasa(split_size, path):
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


def split_data_kbm(split_size, path, save_path):
    num_features = 4
    df = pd.read_csv(f"{path}", sep=',')
    # drop the ending rows not dividable by 4000
    df = df.iloc[:-(len(df) % 4000)]
    print(len(df))
    df_measurements = np.array_split(df, len(df)/4000)
    for df_measurement in df_measurements:
        dfs = np.array_split(df_measurement, split_size)
        for df_chunk in dfs:
            mean_abs = np.array(df_chunk.abs().mean())
            mean_abs = pd.DataFrame(mean_abs.reshape(1, num_features))
            mean_abs.to_csv(save_path, mode='a', index=False, header=False, sep=';')


def import_csv(path, name):
    df = pd.read_csv(path, sep=',')
    df[['tags', 'temperature']] = df['tags'].str.split('temperature=', expand=True)
    # drop tags and time column from dataframe
    df.drop(columns=['tags', 'time'], inplace=True)
    print(df)
    df.to_csv(f"data/kbm_dataset/{name}_4000.csv", index=False)


if __name__ == '__main__':
    split_data_kbm(split_size=100, path='data/kbm_dataset/stabilus_4000.csv',
                   save_path="data/kbm_dataset/stabilus_100.csv")
   #  import_csv('archive/data/KBM Dataset/eval-Pumpe134_v3.csv', name='pumpe_v3')
