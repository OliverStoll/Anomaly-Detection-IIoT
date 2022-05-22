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


class DataPipeKBM:
    def __init__(self, import_path: str, export_path: str):
        self.import_path = import_path
        self.export_path = export_path

    def prepare_data(self, sep=',', split_tags=True):
        """
        Function to clean all csv files of the kbm dataset. The function extracts the relevant columns.

        :param sep: separator of the csv files
        :param split_tags: if True, the tags are split into separate columns
        """

        # iterate over all files from import_path directory
        for file in os.scandir(self.import_path):

            # read original csv file
            df = pd.read_csv(file, sep=',')

            # extract the temperature from the tags column and remove overhang from the tags column
            if split_tags:
                df[['tags', 'temperature']] = df['tags'].str.split('temperature=', expand=True)
                df['temperature'] = df['temperature'].str.split(' ', expand=True)[0]

            # sort the dataframe by time
            df = df.sort_values(by=['time'])

            # add a column with the time in seconds to unify measurements, and a column for anomaly values
            df['time_sec'] = df['time'].apply(lambda x: x.split('.')[0])

            # keep only time and measurement values
            df = df[['vibration-x', 'vibration-y', 'vibration-z', 'temperature', 'time', 'time_sec']]

            # print debug information
            print(df)

            # save cleaned data
            df.to_csv(f"{self.export_path}/{file.name}_full.csv", index=False)

    def check_data_quality(self):

        # iterate over all file names
        for file in os.scandir(self.import_path):

            # read csv file from path
            path = f"{self.export_path}/{file.name}_full.csv"
            df = pd.read_csv(path)

            # check the number of samples per timestamp and timestamps that are directly adjacent
            old_time = datetime.datetime.now() - datetime.timedelta(days=100000)
            for time_str in df['time_sec'].unique():
                time = datetime.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
                time_dif = time - old_time
                old_time = time
                if time_dif < datetime.timedelta(seconds=0):
                    print(file.name, time, time_dif)
                elif time_dif == datetime.timedelta(seconds=1):
                    print(file.name, time, time_dif)


class DataPipeBearingTest:
    """
    Class for cleaning one experiment of the bearing dataset. This can take some time!

    The data is first aggregated and then split into one dataframe per bearing in the experiment.
    Timestamps are added from the file names. Finally the data can be labeled, with anomalies indicated by a 1.
    """

    def __init__(self, export_path, import_path):
        self.export_path = export_path
        self.import_path = import_path
        self.num_bearings = 4
        print(f"DataCleaner: {self.import_path}")

    def prepare_data(self):
        """
        Loads all csv files from the data path and creates a dataframe containing all data from this test-run.

        The files contain one or multiple features from multiple bearings that were being tested simultaneously.
        """

        # get all files from the path and concatenate them to a single csv file
        for file in os.scandir(self.import_path):
            df = pd.read_csv(f"{self.import_path}/{file.name}", sep='\t', header=None)
            df['time_sec'] = datetime.datetime.strptime(file.name, '%Y.%m.%d.%H.%M.%S')
            df.to_csv(self.export_path + "_full.csv", mode='a', index=False, header=False)  # must not exist yet!!

        # read csv file from path and add column names to use the time later
        df = pd.read_csv(self.export_path + "_full.csv", header=None)
        df.columns = ["Vibration-" + str(i) for i in range(df.shape[1] - 1)] + ['time_sec']
        df.to_csv(self.export_path + "_full.csv", index=False)

    def plot_data(self):
        """
        Plots the raw vibration data from the test-run. This allows a visual anotation of the data.

        This method is obviously far from perfect for accurate anomaly detection and is only usable as the data is
        clearly defined as 'run until failure'.
        """

        # read csv files from path
        for i in range(self.num_bearings):
            df = pd.read_csv(f"{self.export_path}_bearing-{i}_original.csv", header=None)
            # create figure
            plt.figure(figsize=(20, 20))
            plt.plot(df)
            for i in range(100):
                # draw vertical lines of the x-axis for visual inspection of anomalies
                if i % 5 == 0:
                    plt.axvline(i * int(df.shape[0] / 100), color='r', linestyle='-')
                else:
                    plt.axvline(i * int(df.shape[0] / 100), color='green', linestyle='-')
            plt.show()


class Resampler:
    def __init__(self, directory_path, new_sampling_rate, feature_cols_tuple):
        self.directory_path = directory_path
        self.new_sampling_rate = new_sampling_rate
        self.feature_cols_tuple = feature_cols_tuple
        self.old_sampling_rate = 20480 if 'bearing' in self.directory_path else 800

    def resample_all_csv_in_directory(self):
        """
        Resample all files in a folder to a new sampling rate.

        The files need to be all of the same type (bearing or kbm) and have the same column names!

        :param new_sampling_rate: number of samples to resample to
        """

        # iterate over all files in folder
        for file in os.listdir(self.directory_path):
            if file.endswith("_full.csv"):
                print(f"Resampling {file} Rate: {self.new_sampling_rate}")
                self.resample_csv(file_path=f"{self.directory_path}/{file.replace('_full.csv', '')}")

    def resample_csv(self, file_path):
        """
        Resample data from a csv file to a new sampling rate.

        :param new_sampling_rate: number of samples to resample to
        """

        # load data from file as dataframe
        df = pd.read_csv(f"{file_path}_full.csv")

        # delete resample file if another already exists
        save_path = f"{file_path}_{self.new_sampling_rate}.csv"
        if os.path.exists(save_path):
            print("Delete old")
            os.remove(save_path)

        # get the number of samples in the data and the column names
        num_samples = len(df) / self.old_sampling_rate
        column_names = df.columns.values.tolist()

        # drop the ending rows not dividable by original sampling rate
        print(len(df) % self.old_sampling_rate)
        if 'kbm' in file_path:
            df = df.iloc[:-(len(df) % self.old_sampling_rate)]
        list_samples_df = np.array_split(df, num_samples)

        print(int(len(df) / (self.old_sampling_rate * 100)) * '█')

        # iterate over all samples
        counter = 0
        for df_sample in list_samples_df:
            counter += 1
            if counter % 100 == 0:
                print('█', end='')

            # split the sample dataframe into as many chunks as the new sampling rate
            dfs = np.array_split(df_sample, self.new_sampling_rate)
            for df_chunk in dfs:
                # reset index of the chunk
                df_chunk.reset_index(drop=True, inplace=True)
                if self.feature_cols_tuple is None:
                    df_features_chunk = df_chunk.iloc[:, :-1]
                else:
                    df_features_chunk = df_chunk.iloc[:, self.feature_cols_tuple[0]:self.feature_cols_tuple[1]]
                # take the mean of each chunk for downsampling and save it to the new file
                mean_abs = np.array(df_features_chunk.abs().mean())
                new_sample = pd.DataFrame(mean_abs.reshape(1, -1))
                new_sample['time_sec'] = df_chunk['time_sec'][0]
                new_sample.to_csv(save_path, mode='a', index=False, header=False)

        df = pd.read_csv(save_path, header=None)
        df.columns = column_names[self.feature_cols_tuple[0]:self.feature_cols_tuple[1]] + ['time_sec']
        print(df)
        df.to_csv(save_path, index=False)


def _resample_all_to_defaults():
    for i in [30, 100, 512]:
        # Resampler(directory_path='data/kbm', new_sampling_rate=i, feature_cols_tuple=(2, 6)).resample_all_csv_in_directory()
        Resampler(directory_path='data/bearing', new_sampling_rate=i, feature_cols_tuple=(0, 4)
                  ).resample_all_csv_in_directory()


if __name__ == '__main__':
    _resample_all_to_defaults()
