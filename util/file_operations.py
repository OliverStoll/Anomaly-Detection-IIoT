import numpy as np
import pandas as pd
import os
import datetime
from matplotlib import pyplot as plt


files_kbm = ["stabilus", "stadtkehl", "wasserwerke-a", "wasserwerke-b", "piaggio", "gummipumpe", "pumpe-v2", "pumpe-v3"]

anomalies_kbm = {'stabilus': ["17/06/19 10:22:00",
                                  "17/06/19 11:22:00",
                                  "17/06/19 12:24:00",
                                  "17/06/19 13:24:00"],
                 'stadtkehl': ["22/05/19 06:39:59"
                               "22/05/19 07:40:36",
                               "22/05/19 08:41:13",
                               "22/05/19 09:41:51"
                               ],
                 'pumpe-v2': ["2/16/18 17:59",
                              "2/16/18 18:59"],
                 'pumpe-v3': ["5/20/18 7:35",
                              "5/25/18 18:35"]}

anomalies_bearing = {'test1': [1.000, 0.985, 0.840, 0.665],
                     'test2': [0.710, 0.980, 1.000, 0.900],
                     'test3': [0.980, 0.975, 0.972, 0.972]}


class Resampler:
    """
    Resampler class for resampling data to a lower sampling frequency.

    The class has two methods for the two datasets used, bearings and kbm.
    """

    def __init__(self, data_path):
        self.data_path = data_path
        self.dataset = "kbm" if "kbm" in data_path else "bearing"

    def resample_defaults(self):
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

    def resample_data_bearing(self, resample_size=20480):
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
                mean_abs = pd.DataFrame(mean_abs.reshape(1, -1))
                # append the new sample to the file
                mean_abs.to_csv(f"{self.data_path}_{resample_size}.csv", mode='a', index=False, header=False, sep=',')

    def resample_data_kbm(self, resample_size, sampling_rate=800):
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
        list_df_measurements = np.array_split(df, len(df) / 4000)

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
                mean_abs = pd.DataFrame(mean_abs.reshape(1, -1))
                mean_abs.to_csv(save_path, mode='a', index=False, header=False, sep=';')


class DataCleanerKBM:
    def __init__(self, file_names: list[str], data_path: str, anomaly_timestamps: dict):
        self.file_names = file_names
        self.data_path = data_path
        self.anomaly_timestamps = anomaly_timestamps

    def clean_data(self, sep=',', split_tags=True):
        """
        Function to clean all csv files of the kbm dataset. The function extracts the relevant columns.

        :param sep: separator of the csv files
        :param split_tags: if True, the tags are split into separate columns
        """

        # iterate over all file names
        for file_name in self.file_names:
            path = f"archive/{self.data_path}/{file_name}.csv"

            # read original csv file from path
            df = pd.read_csv(path, sep=sep)

            # extract the temperature from the tags column
            if split_tags:
                df[['tags', 'temperature']] = df['tags'].str.split('temperature=', expand=True)

            # remove overhang from the tags column
            df['temperature'] = df['temperature'].str.split(' ', expand=True)[0]

            # sort the dataframe by time
            df = df.sort_values(by=['time'])

            # add a column with the time in seconds to unify measurements, and a column for anomaly values
            df['time_sec'] = df['time'].apply(lambda x: x.split('.')[0])
            df['anomaly'] = 0

            # keep only time and measurement values
            df = df[['time', 'time_sec', 'x', 'y', 'z', 'temperature', 'anomaly']]

            # print debug information
            print(df)

            # save cleaned data
            df.to_csv(f"data/kbm_dataset/{file_name}_original.csv", index=False)

    def check_data(self):

        # iterate over all file names
        for file_name in self.file_names:

            # read csv file from path
            path = f"{self.data_path}/{file_name}_original.csv"
            df = pd.read_csv(path)

            # check the number of samples per timestamp and timestamps that are directly adjacent
            old_time = datetime.datetime.now() - datetime.timedelta(days=100000)
            for time_str in df['time_sec'].unique():
                time = datetime.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
                time_dif = time - old_time
                old_time = time
                if time_dif < datetime.timedelta(seconds=0):
                    print(file_name, time, time_dif)
                elif time_dif == datetime.timedelta(seconds=1):
                    print(file_name, time, time_dif)

    def add_label(self, excel_timestamps=True):

        # iterate over all file names
        for file_name in self.file_names:

            # read data from path
            path = f"{self.data_path}/{file_name}"
            df = pd.read_csv(path)
            timestamps = self.anomaly_timestamps[file_name]

            # convert timestamps from excel format (for easy copy pasting from excel sheet)
            if excel_timestamps:
                timestamps = [datetime.datetime.strptime(ts, '%d/%m/%y  %H:%M:%S') for ts in timestamps]

            for timestamp in timestamps:
                df['anomaly'] += df['time'].str.match(str(timestamp)).astype(int)

            # set anomaly values to 1
            df[df['anomaly'] > 0] = 1

            # print debug information
            print(df['anomaly'].value_counts())

            # save cleaned data
            df.to_csv(f"data/kbm_dataset/{file_name}_labeled.csv", index=False)


class DataCleanerBearingTest:
    def __init__(self, data_path, num_features=1):
        self.data_path = data_path
        self.archive_path = "archive/" + self.data_path
        self.num_features = num_features
        self.num_bearings = 4
        print(f"DataCleanerBearing: {self.data_path}")

    def clean_data(self):
        """
        Loads all csv files from the data path and creates a dataframe containing all data from this test-run.

        The files contain one or multiple features from multiple bearings that were being tested simultaneously.
        """

        # get all files from the path and concatenate them to a single csv file
        for file in os.scandir(self.archive_path):
            df = pd.read_csv(f"{self.archive_path}/{file.name}", sep='\t', header=None)
            df.to_csv(self.data_path + "_original.csv", mode='a', index=False, header=False)  # needs to not exist yet!!

        # read csv file from path
        df = pd.read_csv(self.data_path + "_original.csv", header=None)

        # take the created csv file and split it into multiple files containing one bearing.
        for i in range(self.num_bearings):
            print(i)
            df_bearing = df.iloc[:, i * self.num_features:(i + 1) * self.num_features]  # select num_features columns
            # name the columns and save the dataframe
            df_bearing.columns = ["Vibration-" + str(i) for i in range(df_bearing.shape[1])]
            df_bearing.to_csv(f"{self.data_path}_bearing-{i}_original.csv", index=False)

    def plot_data(self):
        """
        Plots the raw vibration data from the test-run. This allows a visual anotation of the data.

        This method is obviously far from perfect for accurate anomaly detection and is only usable as the data is
        clearly defined as 'run until failure'.
        """

        # read csv files from path
        for i in range(self.num_bearings):
            df = pd.read_csv(f"{self.data_path}_bearing-{i}_original.csv", header=None)
            # create figure
            plt.figure(figsize=(20, 20))
            plt.plot(df)
            for i in range(50):
                # draw vertical lines of the x-axis for visual inspection of anomalies
                if i % 5 == 0:
                    plt.axvline(i * int(df.shape[0] / 50), color='green', linestyle='-')
                else:
                    plt.axvline(i * int(df.shape[0] / 50), color='r', linestyle='-')
            plt.show()

    def add_label(self, time_percentile_list):
        """
        Add label if a bearing data is anomalious (yet) or not, where 1 symbolise anomalies.

        Since all tests were 'run until failure', when a bearing is anomalious it will stay this way.
        The time_percentile is used to determine at which point the data is considered anomalius.

        :param time_percentile_list: list of the fraction of the total time at which anomalious behaviour starts.
        """

        assert len(time_percentile_list) == self.num_bearings

        # iterate over all bearing files and add label depending on time
        for i in range(self.num_bearings):
            df = pd.read_csv(f"{self.data_path}_bearing-{i}_original.csv")

            # set all anomaly values to 1 for the second half of the dataframe
            start_index = int(df.shape[0] * time_percentile_list[i])
            df['anomaly'] = 0
            df['anomaly'][start_index:] = 1

            # save the labeled data
            df.to_csv(f"{self.data_path}_bearing-{i}.csv", index=False)


if __name__ == '__main__':
    pass
    # DataCleanerKBM(file_names=files, data_path="data/kbm_dataset", anomaly_timestamps=anomaly_ts).check_data()



    # Resampler(data_path='archive/data/bearing_dataset/test1').resample_data(resample_size=20480)
