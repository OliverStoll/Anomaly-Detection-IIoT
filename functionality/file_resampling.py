import numpy as np
import pandas as pd
import os

from functionality.config import c


def split_data_nasa(split_size, path):
    for file in os.scandir(path):
        df = pd.read_csv(f"{path}/{file.name}", sep='\t', header=None)
        # df.to_csv(f"data/bearing_dataset/absolutes_3_{split_size}.csv", mode='a', index=False, header=False)
        dfs = np.array_split(df, split_size)
        for df_chunk in dfs:
            mean_abs = np.array(df_chunk.abs().mean())
            mean_abs = pd.DataFrame(mean_abs.reshape(1, 4))
            mean_abs.to_csv(f"{path}_{split_size}.csv", mode='a', index=False, header=False)
        print('x', end='')


def split_data_kbm(split_size, path):
    num_features = 4
    save_path = f"{path}_{split_size}.csv"
    # delete old file if exists
    if os.path.exists(save_path):
        os.remove(save_path)
    path = f"{path}_4000.csv"
    df = pd.read_csv(path, sep=',')
    # drop the ending rows not dividable by 4000
    df = df.iloc[:-(len(df) % 4000)]
    print(int(len(df) / 400000) * 'x')
    df_measurements = np.array_split(df, len(df)/4000)

    counter = 0
    for df_measurement in df_measurements:
        counter += 1
        print('x', end='') if counter % 100 == 0 else None
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
    print(len(df))
    df.to_csv(f"data/kbm_dataset/{name}_4000.csv", index=False)


if __name__ == '__main__':
    split_data_kbm(split_size=100, path='data/kbm_dataset/piaggio')