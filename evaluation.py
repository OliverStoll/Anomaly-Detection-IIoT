import numpy as np
import pandas as pd
import os
import yaml
from datetime import datetime
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from util.config import c

mpl.rcParams['agg.path.chunksize'] = 10000

# load yaml anomalies.yaml
anomalies = yaml.safe_load(open(f"files/anomalies.yaml"))
anomalies_bearing = anomalies['bearing']
anomalies_kbm = anomalies['kbm']


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
        plt.plot(results[f'Pred_{i}'], label=f'pred_{i}', color=(1 - i * 0.1, 0, 0))
        plt.plot(results[f'Data_{i}'], label=f'actual_{i}', color=(0, 0, 1 - i * 0.1))
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
        plt.plot(results_chunk[f'Data_{i}'].tolist(), label=f"Data_{i}", color=(1 - i * 0.3, i * 0.3, 0), linestyle='-')
        plt.plot(results_chunk[f'Pred_{i}'].tolist(), label=f"Prediction_{i}", color=(1 - i * 0.3, i * 0.3, 0),
                 linestyle='--')
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

    return list_mse

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

    return mse


class PlotterRaw:
    """
    Class to plot the raw data and anomalies for a given experiment.

    todo: add support for plotting the anomaly score for the lstm model
    """

    def __init__(self, file_path, sub_experiments, features, ylim, anomalies_real: dict = None):
        self.file_path = file_path
        self.sub_experiments = sub_experiments
        self.features = features
        self.ylim = ylim
        self.anomalies_real = anomalies_real

        # load dataframe and create figure
        self.df = pd.read_csv(self.file_path)
        self.x_max = self.df.shape[0]
        self.fig = plt.figure(figsize=(12, 12))
        if sub_experiments > 1:
            self.outer = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2)
        else:
            self.outer = gridspec.GridSpec(1, 1, wspace=0.2, hspace=0.2)

    def plot_experiment(self):
        """
        Higher level function to plot the whole experiment.

        For every sub experiment, a subplot is created.
        """

        # load the data
        print(f"Plotting {self.file_path.split('/')[-1]}")

        # iterate over the bearing index to plot all bearings in one figure
        for sub_experiment in range(self.sub_experiments):

            # plot each bearing as a subplot, with multiple features as gridspecs
            inner = gridspec.GridSpecFromSubplotSpec(self.features, 1, subplot_spec=self.outer[sub_experiment],
                                                     wspace=0.1, hspace=0.1)
            for index in range(self.features):
                self._add_single_subplot(inner=inner, sub_experiment=sub_experiment, index=index)

        # plot title and show the figure
        self.fig.suptitle(f"Raw Vibration Data: {self.file_path.split('/')[-1]}", fontsize=20)
        self.fig.show()

    def _add_single_subplot(self, inner, sub_experiment, index):
        """
        Function that plots the original data. This is used to visualize the data and show anomalies.

        The anomalies are plotted as red colored background.
        """
        ax = plt.Subplot(self.fig, inner[index])
        sub_df = self.df.iloc[:, index + sub_experiment * self.features]

        ax.plot(sub_df.abs(), label=self.df.columns[index])
        if index == 0:
            ax.set_title(f"Experiment {sub_experiment}")

        # if ylim was given as list, apply each ylim individually
        if isinstance(self.ylim, list):
            ylim = self.ylim[index]
        else:
            ylim = self.ylim

        if isinstance(ylim, tuple):
            ylim_min, ylim_max = ylim
            ax.set_yticks([ylim_min, ylim_max], [ylim_min, ylim_max])
        else:
            ylim_min = -0.05*ylim
            ylim_max = ylim
            ax.set_yticks([0, ylim_max], [0, ylim_max])

        ax.set_xticks([0, self.x_max], ['start', 'end'])
        ax.set_xlabel('Time')

        # get the column name of the current feature
        column_name = self.df.columns[index + sub_experiment * self.features]
        ax.set_ylabel(column_name)

        # plot the label anomalies if available
        if self.anomalies_real is not None:
            anomalies_real = list(self.anomalies_real.items())[sub_experiment][1]
            self._add_anomaly_list_to_subplot(ax=ax, color='red', anomaly_list=anomalies_real)

        self.fig.add_subplot(ax)

    def _add_anomaly_list_to_subplot(self, ax, color, anomaly_list):
        for anomaly_data in anomaly_list:
            if isinstance(anomaly_data, tuple) or isinstance(anomaly_data, list):  # check if anomaly data is tuple or single value
                # plot vertical area
                ax.axvspan(anomaly_data[0] * self.x_max, anomaly_data[1] * self.x_max, alpha=0.5, color=color)
            else:
                # plot vertical line if anomaly is a single value
                ax.axvline(anomaly_data * self.x_max, color=color, alpha=0.5)


def get_timestamp_percentiles(path, timestamps):
    """
    Function that returns the index of the timestamp in the given file.
    """

    list_percentiles = []
    for timestamp in timestamps:
        # convert timestamp to datetime object if needed
        try:  # normal timestamp
            timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
        except:
            timestamp = datetime.strptime(timestamp, '%d/%m/%Y  %H:%M:%S')

        df = pd.read_csv(path)
        time_series = df['time_sec']

        # find first index where timestamp is greater than the time_series
        index = time_series.searchsorted(str(timestamp))
        max = len(time_series)
        list_percentiles.append(index / max)
    return list_percentiles


def _plot_kbm(ending):
    # iterate over all files in data/kbm
    for file in os.listdir('data/kbm'):
        # plot if file is a csv file
        if file.endswith(ending):
            PlotterRaw(file_path=f'data/kbm/{file}', sub_experiments=1, features=4,
                       anomalies_real=anomalies_kbm[file.split("/")[-1].replace(ending, "")], ylim=[500, 500, 1500, 50]
                       ).plot_experiment()


def _plot_bearing(ending):
    # iterate over all files in data/bearing
    for file in os.listdir('data/bearing'):
        # plot if file is a csv file
        if file.endswith(ending):
            # hardcode number of features
            features = 2 if 'experiment-1' in file else 1
            PlotterRaw(file_path=f'data/bearing/{file}', sub_experiments=4, features=features,
                       anomalies_real=anomalies_bearing[file.split("/")[-1].replace(ending, "")], ylim=.5
                       ).plot_experiment()


if __name__ == '__main__':
    _plot_bearing(ending='_10.csv')




