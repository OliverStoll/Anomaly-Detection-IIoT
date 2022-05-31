import numpy as np
import pandas as pd
import os
import yaml
from datetime import datetime
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from util.config import c

# plot big images
mpl.rcParams['agg.path.chunksize'] = 10000

# load anomalies dict from anomalies.yaml
anomalies = yaml.safe_load(open(f"files/anomalies.yaml"))


class MiscPlotter:
    def __init__(self, trainer):
        self.trainer = trainer

    def plot_infotable(self):
        """
        Plot the predictions and the actual values of the dataset. Also plot all relevant metrics such as loss and val_loss.

        :param trainer: the trainer object that contains the model and the data
        :return:
        """

        # extract the relevant data from the trainer object
        trainer = self.trainer
        data = trainer.data_2d[:, 0]
        pred = trainer.data_pred_2d[:, 0]
        mse = trainer.mse_lstm
        loss = trainer.history_lstm['loss']
        val_loss = trainer.history_lstm['val_loss']
        num_features = len(trainer.data_columns)

        results = pd.DataFrame()
        results['Pred_0'] = pred
        results['Data_0'] = data

        # take a chunk of the size of three measurements and plot the predictions and the actual values
        results_chunk = results.iloc[50 * c.SPLIT:53 * c.SPLIT, :]

        # create figure
        plt.figure(figsize=(15, 10))
        font_d = {'size': 12}

        # subplot the predictions and the actual values
        plt.subplot(2, 2, 1)
        for i in range(1):  # only plots one feature
            plt.plot(results[f'Pred_{i}'], label=f'pred_{i}', color=(1 - i * 0.1, 0, 0))
            plt.plot(results[f'Data_{i}'], label=f'actual_{i}', color=(0, 0, 1 - i * 0.1))
            plt.ylim(-0.1, 3)
        plt.legend(loc='upper left')
        plt.title('Predictions and Actual Values', fontdict=font_d)

        # subplot the anomaly score and threshold
        plt.subplot(2, 2, 2)
        plt.title("Anomaly Score", fontdict=font_d)
        plt.plot(mse, label='Loss_MSE')
        plt.ylim(0, min(max(mse) * 1.2, 3 * c.THRESHOLD_LSTM))
        plt.axhline(y=c.THRESHOLD_LSTM, color='r', linestyle='-', label="Threshold")

        # subplot the predictions and the actual values of the chunk
        plt.subplot(2, 2, 3)
        plt.title("Vibration - Chunk", fontdict=font_d)
        for i in range(1):
            plt.plot(results_chunk[f'Data_{i}'].tolist(), label=f"Data_{i}", color=(1 - i * 0.3, i * 0.3, 0), linestyle='-')
            plt.plot(results_chunk[f'Pred_{i}'].tolist(), label=f"Prediction_{i}", color=(1 - i * 0.3, i * 0.3, 0),
                     linestyle='--')
        plt.legend()

        # subplot all metrics in text form
        plt.subplot(2, 2, 4)
        plt.axis('off')
        if len(loss) > 0 and len(val_loss) > 0:
            text = f"Epochs: {c.EPOCHS}\nLoss:       {loss[-1]:.1e}\nVal_Loss: {val_loss[-1]:.1e}\n\n" \
                   f"Batch: {c.BATCH_SIZE}\nLearn_r: {c.LEARNING_RATE:.0e}\nLR_Red: {c.LR_DECAY}" \
                   f"\n\nLSTM: {c.OUTER_LAYER_SIZE}-{c.HIDDEN_LAYER_SIZE}\nSplit: {c.SPLIT}"
            plt.text(0.2, 0.9, text, ha='left', va='top', fontdict={'fontsize': 20})

        # show the figure
        plt.suptitle(f"Infotable {trainer.experiment_name} ({os.getenv('CLIENT_NAME')})", fontsize=20)
        plt.show()

    def plot_anomaly_scores(self):

        # get the mses
        mse_lstm = self.trainer.mse_lstm
        mse_fft = self.trainer.mse_fft

        # add two subplots
        fig, ax = plt.subplots(2, 1, figsize=(15, 10))

        # plot the mse of the lstm model
        ax[0].plot(mse_lstm)
        ax[0].set_title("LSTM Autoencoder Anomaly Score")
        ax[0].set_ylim(0, 3 * c.THRESHOLD_LSTM)
        ax[0].axhline(y=c.THRESHOLD_LSTM, color='r', linestyle='-')  # plot horizontal line at threshold

        # plot the mse of the fft model
        ax[1].plot(mse_fft)
        ax[1].set_title("FFT Autoencoder Anomaly Score")
        ax[1].set_ylim(0, 3 * c.THRESHOLD_FFT)
        ax[1].axhline(y=c.THRESHOLD_FFT, color='r', linestyle='-')  # plot horizontal line at threshold

        # show the plot
        plt.suptitle(f"Anomaly Scores ({os.getenv('CLIENT_NAME')})", fontsize=20)
        fig.show()

    def plot_losses(self, ylim):
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 1, 1)
        plt.ylim([0, ylim])
        plt.plot(self.trainer.history_lstm['loss'], label='Loss')
        plt.plot(self.trainer.history_lstm['val_loss'], label='Val_Loss')
        plt.title("LSTM Autoencoder Loss")
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.ylim([0, ylim])
        plt.plot(self.trainer.history_fft['loss'], label='Loss')
        plt.plot(self.trainer.history_fft['val_loss'], label='Val_Loss')
        plt.title("FFT Autoencoder Loss")
        plt.legend()
        plt.suptitle(f"Losses ({os.getenv('CLIENT_NAME')})", fontsize=20)
        plt.show()

    def plot_roc(self):
        roc_plotter = RocPlotter()


class RocPlotter:
    def __init__(self):
        self.fig = plt.figure(figsize=(15, 10))
        self.fig.suptitle(f"ROC Curve ({os.getenv('CLIENT_NAME')})", fontsize=20)
        self.subplot_x = 2
        self.subplot_index = 1

    def show(self):
        self.fig.show()

    def plot_single_roc(self, fps, tps, auc, f1_max):

        ax = self.fig.add_subplot(1, self.subplot_x, self.subplot_index)
        self.subplot_index += 1
        ax.plot(fps[f1_max[1]], tps[f1_max[1]], 'ro', label='Max F1 Score')

        fps = np.insert(fps, 0, 0)
        tps = np.insert(tps, 0, 0)

        ax.plot(fps, tps, label='ROC curve')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([-0.01, 1.0])
        ax.set_ylim([0.0, 1.01])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'AUC={auc:.4f}   (F1={f1_max[0]:.4f})', fontsize=15)
        ax.legend(loc="upper right")
        return ax


class PredictionsPlotter:
    """
    Class to plot the raw data and anomalies for a given experiment.

    todo: add support for plotting the anomaly score for the lstm model
    """

    def __init__(self, file_path, features, ylim, sub_plots=1, sub_experiment_index=0, anomalies_real: dict = None,
                 anomalies_pred_lstm: list = None, anomalies_pred_fft: list = None):
        self.file_path = file_path
        self.sub_plots = sub_plots
        self.sub_experiments_index = sub_experiment_index
        self.features = features
        self.ylim = ylim
        self.anomalies_real = anomalies_real
        self.anomalies_pred_lstm = anomalies_pred_lstm
        self.anomalies_pred_fft = anomalies_pred_fft

        # load dataframe and create figure
        self.df = pd.read_csv(self.file_path)
        self.x_max = self.df.shape[0]
        self.fig = plt.figure(figsize=(15, 10))
        if sub_plots > 1:
            self.outer = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2)
        else:
            self.outer = gridspec.GridSpec(1, 1, wspace=0.2, hspace=0.2)

        assert not (anomalies_pred_lstm and (sub_plots > 1)), "Only predict anomalies for single experiment"
        # todo: add support for plotting the anomalies for multiple sub-experiments

    def plot_experiment(self):
        """
        Higher level function to plot the whole experiment.

        For every sub experiment, a subplot is created.
        """

        # load the data
        print(f"Plotting {self.file_path.split('/')[-1]}")

        # iterate over the bearing index to plot all bearings in one figure
        for sub_experiment in range(self.sub_plots):

            # plot each bearing as a subplot, with multiple features as gridspecs
            inner = gridspec.GridSpecFromSubplotSpec(self.features, 1, subplot_spec=self.outer[sub_experiment],
                                                     wspace=0.1, hspace=0.1)
            for index in range(self.features):
                self._add_single_subplot(inner=inner, sub_experiment=sub_experiment, index=index)

        # plot title and show the figure
        self.fig.suptitle(f"Predictions: {self.file_path.split('/')[-1].replace('.csv', '')} ({os.getenv('CLIENT_NAME')})",
                          fontsize=20)
        self._add_legend_to_figure()
        self.fig.show()

    def _add_single_subplot(self, inner, sub_experiment, index):
        """
        Function that plots one feature. This is used to visualize the data and show anomalies.

        The anomalies are plotted as red colored background.
        """

        ax = plt.Subplot(self.fig, inner[index])
        sub_df = self.df.iloc[:, index + sub_experiment * self.features + self.sub_experiments_index * self.features]

        ax.plot(sub_df.abs(), label=self.df.columns[index])
        if index == 0:
            ax.set_title(f"Experiment {sub_experiment + self.sub_experiments_index}")

        # if ylim was given as list, apply the right ylim for this subexperiment
        if isinstance(self.ylim, list):
            ylim_subplot = self.ylim[index]
        else:
            ylim_subplot = self.ylim

        # if ylim for subplot is tuple (list), use ylim-min additionaly to ylim-max
        if isinstance(ylim_subplot, list):
            ylim_min, ylim_max = ylim_subplot
            ax.set_yticks([ylim_min, ylim_max], [ylim_min, ylim_max])
        else:
            ylim_min = -0.05 * ylim_subplot
            ylim_max = ylim_subplot
            ax.set_yticks([0, ylim_max], [0, ylim_max])
        ax.set_ylim(ylim_min, ylim_max)

        ax.set_xticks([0, self.x_max], ['start', 'end'])
        ax.set_xlabel('Time')

        # get the column name of the current feature
        column_name = self.df.columns[index + sub_experiment * self.features]
        ax.set_ylabel(column_name)

        # plot the label anomalies if available
        if self.anomalies_real and len(self.anomalies_real) > 0:
            anomalies_real = list(self.anomalies_real.items())[sub_experiment+self.sub_experiments_index][1]  # anomalies real are whole yaml dict
            minmax = (0.05, 0.25) if self.anomalies_pred_lstm else (0.05, 0.95)
            self._add_anomalies_to_subplot(ax=ax, color='red', anomaly_list=anomalies_real, minmax=minmax)

        # plot the predicted anomalies if available
        if self.anomalies_pred_lstm:
            self._add_anomalies_to_subplot(ax=ax, color='purple', anomaly_list=self.anomalies_pred_lstm, minmax=(0.675, 0.95))
        if self.anomalies_pred_fft:
            self._add_anomalies_to_subplot(ax=ax, color='green', anomaly_list=self.anomalies_pred_fft, minmax=(0.35, 0.625))

        # plot the training split as vertical line, if training data was split
        if c.PLOT_SPLITS:
            ax.axvline(x=c.TRAIN_SPLIT * self.x_max, color='black', alpha=0.5, linestyle='-')
            ax.axvline(x=c.TRAIN_SPLIT * (1-c.VAL_SPLIT) * self.x_max, color='black', alpha=0.5, linestyle='--')

        self.fig.add_subplot(ax)

    def _add_anomalies_to_subplot(self, ax, color, anomaly_list, minmax=(0, 1)):
        for anomaly_data in anomaly_list:
            if isinstance(anomaly_data, list):  # check if anomaly data is tuple or single value
                # plot vertical area
                ax.axvspan(anomaly_data[0] * self.x_max, anomaly_data[1] * self.x_max, alpha=0.5, color=color,
                           ymin=minmax[0], ymax=minmax[1])
            else:
                # plot vertical line if anomaly is a single value
                ax.axvline(anomaly_data * self.x_max, color=color, alpha=0.5)

    def _add_legend_to_figure(self):
        custom_lines = [Line2D([0], [0], color='purple', lw=4),
                        Line2D([0], [0], color='green', lw=4),
                        Line2D([0], [0], color='red', lw=4)]
        split_lines = [Line2D([0], [0], color='black', linestyle='-', lw=4),
                        Line2D([0], [0], color='black', linestyle='--', lw=4)]
        titles = ['Anomalies LSTM', 'Anomalies FFT', 'Labeled']
        split_titles = ['Training Split', 'Validation Split']
        if c.PLOT_SPLITS:
            custom_lines += split_lines
            titles += split_titles
        leg = self.fig.legend(custom_lines,
                              titles,
                              loc='upper right')
        for lh in leg.legendHandles:
            lh.set_alpha(0.5)


def _plot_all_kbm(ending):
    # iterate over all files in data/kbm
    for file in os.listdir('data/kbm'):
        # plot if file is a csv file
        if file.endswith(ending):
            PredictionsPlotter(file_path=f'data/kbm/{file}', sub_plots=1, features=4,
                               anomalies_real=anomalies['kbm'][file.split("/")[-1].replace(ending, "")], ylim=[500, 500, 1500, 50]
                               ).plot_experiment()


def _plot_all_bearing(ending):
    # iterate over all files in data/bearing
    for file in os.listdir('data/bearing'):
        # plot if file is a csv file
        if file.endswith(ending):
            # hardcode number of features
            features = 2 if 'experiment-1' in file else 1
            PredictionsPlotter(file_path=f'data/bearing/{file}', sub_plots=4, features=features,
                               anomalies_real=anomalies['bearing'][file.split("/")[-1].replace(ending, "")], ylim=.5
                               ).plot_experiment()


if __name__ == '__main__':
    plotter = PredictionsPlotter(file_path=f"data/bearing/experiment-2_10.csv", sub_plots=1, sub_experiment_index=1,
                                 features=1, ylim=.3, anomalies_real=anomalies['bearing']['experiment-2'])
    plotter.plot_experiment()
    #_plot_bearing(ending='_10.csv')
    #_plot_kbm(ending='_10.csv')
