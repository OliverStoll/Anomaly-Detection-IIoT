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

    def resample_data_kbm(self, resample_size, sampling_rate=4000):  # todo: use timestamps instead of sampling rate
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


def clean_csv_kbm(path, name, sep, split_tags=True):
    """
    Function to clean a csv file of the kbm dataset. The function extracts the relevant columns.

    :param path: path to the csv file
    """

    # read original csv file from path
    df = pd.read_csv(path, sep=sep)

    # extract the temperature from the tags column
    if split_tags:
        df[['tags', 'temperature']] = df['tags'].str.split('temperature=', expand=True)

    # remove overhang from the tags column
    df['temperature'] = df['temperature'].str.split(' ', expand=True)[0]

    # keep only time and measurement values
    df = df[['time', 'x', 'y', 'z', 'temperature']]

    print(len(df))
    print(df.head())

    # save cleaned data
    df.to_csv(f"data/kbm_dataset/{name}_4000.csv", index=False)


def check_samples(file):
    df = pd.read_csv(f"data/kbm_dataset/{file}_4000.csv")
    df['time_measure'] = df['time'].apply(lambda x: x.split('.')[0])
    old_time = datetime.datetime.now()
    for time_str in df['time_measure'].unique():
        time = datetime.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
        time_dif = time - old_time
        old_time = time
        if time_dif < datetime.timedelta(seconds=0):
            print(time, time_dif)
        elif time_dif < datetime.timedelta(seconds=100):
            print(time, time_dif)


def clean_samples_kbm(file):
    df = pd.read_csv(f"data/kbm_dataset/{file}_4000.csv")
    # sort by time
    df = df.sort_values(by=['time'])
    df['time_measure'] = df['time'].apply(lambda x: x.split('.')[0])
    df.to_csv(f"data/kbm_dataset/{file}_800.csv", index=False)


def add_label_kbm(file, timestamps, clean_timestamps=True):
    # convert timestamps from excel format (for easy copy pasting from excel sheet)
    if clean_timestamps:
        timestamps = [datetime.datetime.strptime(ts, '%d/%m/%y  %H:%M:%S') for ts in timestamps]

    # load unlabeled data from path
    df = pd.read_csv(f"data/kbm_dataset/{file}_4000.csv")
    df['anomaly'] = 0
    times = df['time'].tolist()


    # iterate over all times
    for timestamp in timestamps:
        # find the nearest time greater than if no exact match
        for t in times:
            # convert comparison-time to datetime
            try:
                t = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f')
            except:
                t = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
            # check if comparison-time is equal or greater than timestamp
            if t == timestamp:
                print(f"EXACT TS: {timestamp}")
                break
            if t > timestamp:
                print(f"REPLACING TS: {timestamp} -> {t}")
                timestamp = t
                break

        # add anomaly label column with 1 when time contains an anomaly timestamp and 0 otherwise
        df['anomaly'] += df['time'].str.match(str(timestamp)).astype(int)

    df[df['anomaly'] > 0] = 1
    print(df['anomaly'].value_counts())
    df.to_csv(f"data/kbm_dataset/{file}labeled_4000.csv", index=False)


if __name__ == '__main__':

    # clean_csv_kbm(path="archive/data/KBM Dataset/New folder/abc-Piaggio/data.csv", name="piaggio", sep=',')

    files = ["stabilus", "stadtkehl", "wasserwerke-a", "wasserwerke-b", "piaggio"]  # "gummipumpe"

    for file in files:
        print(f"\n{file}")
        clean_samples_kbm(file)
        # add_label_kbm(file=file, timestamps=timestamps)

    # Resampler(data_path='data/kbm_dataset/pumpe_v3', num_features=5).resample_default()


