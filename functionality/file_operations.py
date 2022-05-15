import numpy as np
import pandas as pd
import os
import datetime


class Resampler:
    """
    Resampler class for resampling data to a lower sampling frequency.

    The class has two methods for the two datasets used, bearings and kbm.
    """
    def __init__(self, data_path, num_features):
        self.data_path = data_path
        self.dataset = "kbm" if "kbm" in data_path else "bearing"
        self.num_features = num_features

    def resample_default(self):
        # resample data to default sampling sizes
        for resample_size in [10, 30, 100, 300, 1000]:
            self.resample_data(resample_size=resample_size)

    def resample_data(self, resample_size):
        print(f"Resampling {self.data_path} to {resample_size}")

        # delete file for resampled data if it exists
        if os.path.exists(f"{self.data_path}_{resample_size}.csv"):
            os.remove(f"{self.data_path}_{resample_size}.csv")

        # resample data depending on dataset
        if self.dataset == "kbm":
            self.resample_data_kbm(resample_size)
        else:
            self.resample_data_bearing(resample_size)

    def resample_data_bearing(self, resample_size):
        """
        Resample data from KBM dataset.

        :param resample_size: number of samples to resample to
        """

        # iterate over all files in data_path directory
        for file in os.scandir(self.data_path):
            df = pd.read_csv(f"{self.data_path}/{file.name}", sep='\t', header=None)
            # split the data from one file (1s) to resample_size number of dataframes
            dfs = np.array_split(df, resample_size)
            for df_chunk in dfs:
                # each new sample is down-sampled by taking the mean of the dataframe
                mean_abs = np.array(df_chunk.abs().mean())
                mean_abs = pd.DataFrame(mean_abs.reshape(1, 4))
                # append the new sample to the file
                mean_abs.to_csv(f"{self.data_path}_{resample_size}.csv", mode='a', index=False, header=False, sep=';')

    def resample_data_kbm(self, resample_size, sampling_rate=4000):
        """
        Resample data from KBM dataset.

        :param resample_size: number of samples to resample to
        """

        assert resample_size != sampling_rate, "Sampling rate must be different from resample size"

        # delete old file if exists
        save_path = f"{self.data_path}_{resample_size}.csv"
        if os.path.exists(save_path):
            os.remove(save_path)

        # load data from file as dataframe
        df = pd.read_csv(f"{self.data_path}_4000.csv", sep=',')

        # drop the ending rows not dividable by original sampling rate, and create list of all measurements
        df = df.iloc[:-(len(df) % sampling_rate)]
        list_df_measurements = np.array_split(df, len(df)/4000)

        print(int(len(df) / 400000) * '█')
        counter = 0
        # iterate over all measurements
        for df_measurement in list_df_measurements:
            counter += 1
            print('█', end='') if counter % 100 == 0 else None
            # resample each measurement to resample_size
            dfs = np.array_split(df_measurement, resample_size)
            for df_chunk in dfs:
                # down-sample the measurement by taking the mean of each chunk
                mean_abs = np.array(df_chunk.abs().mean())
                mean_abs = pd.DataFrame(mean_abs.reshape(1, self.num_features))
                mean_abs.to_csv(save_path, mode='a', index=False, header=False, sep=';')


def clean_csv_kbm(path, name, sep):
    """
    Function to clean a csv file of the kbm dataset. The function extracts the relevant columns.

    :param path: path to the csv file
    """

    # read original csv file from path
    df = pd.read_csv(path, sep=sep)

    # extract the temperature from the tags column
    df[['tags', 'temperature']] = df['tags'].str.split('temperature=', expand=True)

    # keep only time and measurement values
    df = df[['time', 'x', 'y', 'z', 'temperature']]

    print(len(df))
    print(df.head())

    # save cleaned data
    df.to_csv(f"data/kbm_dataset/{name}_4000.csv", index=False)


def add_anomaly_data_kbm(path, timestamps, clean_timestamps=True):
    # convert timestamps from excel format (for easy copy pasting from excel sheet)
    if clean_timestamps:
        timestamps = [str(datetime.datetime.strptime(ts, '%d/%m/%Y  %H:%M:%S')) for ts in timestamps]

    print(timestamps)
    df = pd.read_csv(path)

    # add anomaly label column with 1 when time contains an anomaly timestamp and 0 otherwise
    df['anomaly'] = 0
    for timestamp in timestamps:
        df['anomaly'] += df['time'].str.match(timestamp).astype(int)

    print(df['anomaly'].value_counts())
    print(df['time'])


if __name__ == '__main__':
    # clean_csv_kbm("archive/data/KBM Dataset/abc-WasserWerkeBliestalP1A_birchfft_v10.csv", "wasserwerke-a", sep=',')

    # todo: find nearest timestamp!! (anomaly timestamps are wrong)
    timestamps = ["22/05/2019  06:39:59"]
    add_anomaly_data_kbm("data/kbm_dataset/stadtkehl_4000.csv", timestamps=timestamps, clean_timestamps=False)

    # Resampler(data_path='data/kbm_dataset/pumpe_v3', num_features=5).resample_default()


