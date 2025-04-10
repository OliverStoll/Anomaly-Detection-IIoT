import datetime
import os
import pandas as pd
from matplotlib import pyplot as plt


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
        for file in os.scandir(self.import_path):
            df = pd.read_csv(f"{self.import_path}/{file.name}", sep='\t', header=None)
            df['time_sec'] = datetime.datetime.strptime(file.name, '%Y.%m.%d.%H.%M.%S')
            df.to_csv(self.export_path + "_full.csv", mode='a', index=False, header=False)  # must not exist yet!!
        df = pd.read_csv(self.export_path + "_full.csv", header=None)
        df.columns = ["Vibration-" + str(i) for i in range(df.shape[1] - 1)] + ['time_sec']
        df.to_csv(self.export_path + "_full.csv", index=False)

    def plot_data(self, fig_size=(20, 20)):
        """
        Plots the raw vibration data from the test-run. This allows a visual anotation of the data.

        This method is obviously far from perfect for accurate anomaly detection and is only usable as the data is
        clearly defined as 'run until failure'.
        """
        for i in range(self.num_bearings):
            df = pd.read_csv(f"{self.export_path}_bearing-{i}_original.csv", header=None)
            plt.figure(figsize=fig_size)
            plt.plot(df)
            for i in range(100):
                if i % 5 == 0:
                    plt.axvline(i * int(df.shape[0] / 100), color='r', linestyle='-')
                else:
                    plt.axvline(i * int(df.shape[0] / 100), color='green', linestyle='-')
            plt.show()
