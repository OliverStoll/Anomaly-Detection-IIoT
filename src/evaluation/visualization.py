import numpy as np
import pandas as pd
import os
import yaml
import json
from datetime import datetime
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import colors
from sklearn.preprocessing import StandardScaler
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_squared_error

from config.config import config

# plot big images
mpl.rcParams['agg.path.chunksize'] = 10000

anomal_bearings = {'bearing_experiment-1': [2, 3], 'bearing_experiment-2': [0], 'bearing_experiment-3': [3]}
resource_dict = {'runtime': [2413, 6088, 3262], 'cpu_capped_epoch_time': [191.4, 605.2, 485.2], 'cpu': [0.495746727,	0.621289861, 0.92958637], 'bullshit': []}
resource_dict_2 = {'cpu_capped_runtime': [1196.2, 3782.4, 3032.3], 'memory_mean': [0.695, 0.909, 8.340], 'memory_max': [0.769, 0.929, 11.241]}
mem_resources = [[0.695, 0.909, 8.340], [0.769, 0.929, 11.241]]


class Plotter:
    def __init__(self, experiment_name="bearing_experiment-2", columns=None, rolling_min=1, models=['baseline', 'centralized', 'federated']):
        print(f"PLOTTING: {experiment_name}")
        combine_federated_json(experiment_name)
        self.num_bearings = 4
        self.experiment_name = experiment_name
        self.models = models
        self.results = {'baseline': {}, 'centralized': {}, 'federated': {}, 'federated_transfer': {}}
        self.mse = {'baseline': pd.DataFrame(), 'centralized': pd.DataFrame(), 'federated': pd.DataFrame()}
        self.mse_val = {'baseline': pd.DataFrame(), 'centralized': pd.DataFrame(), 'federated': pd.DataFrame()}
        self.mse_period = {'baseline': pd.DataFrame(), 'centralized': pd.DataFrame(), 'federated': pd.DataFrame()}
        self.threshold = {'baseline': [0.], 'centralized': [0.], 'federated': [0.]}
        self.anomalies = {'baseline': [0], 'centralized': [0], 'federated': [0]}
        self.anomalies_period = {'baseline': [0], 'centralized': [0], 'federated': [0]}
        for model in self.models:
            path = f"logs/{experiment_name}/{model}.json"
            with open(path, 'r') as f:
                self.results[model] = json.load(f)
        self.resource_results = resource_dict
        self.figsize = (10, 7)
        self.resource_types = ['cpu', 'memory_mean', 'memory_max', 'runtime']
        self.data_1000 = pd.read_csv(f"data/{experiment_name}/1000.csv")
        scaler = StandardScaler()
        self.data_1000 = pd.DataFrame(scaler.fit_transform(self.data_1000))
        self.data_full = pd.read_csv(f"data/{experiment_name}/full.csv")
        self.columns = columns if columns is not None else [0, 1, 2, 3]
        self.rolling_min = rolling_min
        self._load_mse_and_calc_anoms(rolling_min=rolling_min)

        print(f"Thresholds: {self.threshold}")
        print(f"Anomaly start: {self.anomalies_period}")

    def _load_mse_and_calc_anoms(self, rolling_min):
        """
        group the results by the measurement period and calculate the mean of the reconstruction error using numpy
        """

        if 'baseline' in self.models:
            mse = np.array(self.results['baseline']['lstm']['mse'])
            mse = mse[:len(mse)-len(mse) % self.num_bearings]
            mse = mse.reshape(-1, self.num_bearings, order='F')
            mse = np.concatenate([mse[:20], mse[:-20]])  # baseline temporalize reihenfolge (lookback)
            self.mse['baseline'] = pd.DataFrame(mse)
        if 'centralized' in self.models:
            self.mse['centralized'] = np.array(self.results['centralized']['lstm']['mse']).reshape(-1, self.num_bearings, order='F')
            self.mse['centralized'] = pd.DataFrame(self.mse['centralized'])
        if 'federated' in self.models:
            self.mse['federated'] = np.array([self.results['federated'][str(i)]['lstm']['mse'] for i in range(self.num_bearings)]).T
            self.mse['federated'] = pd.DataFrame(self.mse['federated'])
        if 'federated_transfer' in self.models:
            self.mse['federated_transfer'] = np.array([self.results['federated_transfer'][str(i)]['lstm']['mse'] for i in range(self.num_bearings)]).T
            self.mse['federated_transfer'] = pd.DataFrame(self.mse['federated_transfer'])

        for model in self.models:
            self.mse_val[model] = self.mse[model].apply(lambda x: x[:int(len(x) / 10)])
            self.threshold[model] = self.mse_val[model].mean() + self.mse_val[model].std() * c.THRESHOLD_STD
            self.mse[model] = self.mse[model].rolling(window=rolling_min).min() if model != 'baseline' else self.mse[model]
            self.threshold[model] = self.threshold[model] * 1.7 if model == 'baseline' else self.threshold[model]
            self.mse_period[model] = self.mse[model].copy()
            repeat_factor = 20480 if model == "baseline" else 1000
            self.mse[model] = self.mse[model].apply(lambda x: np.repeat(x, repeat_factor))
            self.anomalies[model] = [0] * 4
            self.anomalies_period[model] = [0] * 4
            for i in self.columns:
                # find first value where mse_period is above threshold
                re_values = self.mse[model][i].to_numpy()
                threshold = self.threshold[model][i]
                self.anomalies[model][i] = np.argmax(re_values > threshold)
                self.anomalies_period[model][i] = self.anomalies[model][i] // repeat_factor

    def plot_data(self, ylim=.5, plot_split=False):
        """ Plot the raw data and the Reconstruction Error """
        plt.figure(figsize=self.figsize)
        for idx, i in enumerate(self.columns):
            if not plot_split:
                plt.subplot(len(self.columns), 1, idx + 1)

            # plot the raw data
            plt.plot(self.data_1000[i], linewidth=0.2)

            # set the x and y limits and labels
            plt.ylabel(f'Bearing {i}')
            plt.ylim(-ylim, ylim)
            # plt.yticks([])
            # plt.xticks([0, len(self.data_1000)], ["start", "end"])

            # create more plots for single bearing plots
            if plot_split:
                plt.xlabel("Time")
                plt.title(f"Bearing {i}")
                plt.show()
                plt.figure(figsize=self.figsize)
        plt.xlabel('Measurement period of experiment')
        plt.suptitle(f'Vibration Data')
        plt.show() if not plot_split else None

    def plot_period(self, chunk_numbers=[200], mean_window=5, line_width=1.1, ylim=4.5):
        for model in ['centralized']:#, 'federated']:
            # load json data
            title = ["Normal Period", "Abnormal Period"]
            bearings_num = 4 if model != 'federated' else 1
            prediction = pd.read_json(f'logs/{self.experiment_name}/{model}_predictions.json').values
            prediction = [pred[0][0] for pred in prediction]
            prediction = np.array(prediction).reshape(bearings_num, -1, 1000)
            # prediction = prediction * 0.02
            for chunk_number in chunk_numbers:
                # read the data chunk
                data_chunk = self.data_1000.iloc[chunk_number*c.SPLIT:chunk_number*c.SPLIT+c.SPLIT]
                data_chunk = data_chunk.rolling(window=mean_window).mean().to_numpy()

                # plot the raw data
                plt.figure(figsize=self.figsize)
                plt.suptitle(f"Original Input compared to Predicted Output")
                for idx, i in enumerate(self.columns):
                    pred_chunk = prediction[i][int(chunk_number)]
                    pred_chunk = pd.DataFrame(pred_chunk).rolling(window=mean_window).mean().to_numpy()

                    plt.subplot(len(self.columns), 1, idx + 1)
                    plt.plot(data_chunk[:, i], linewidth=line_width*0.5, label='Data', alpha=0.8)
                    plt.plot(pred_chunk, linewidth=line_width, label='Reconstruction')
                    plt.ylim(-ylim, ylim)
                    plt.ylabel(title[idx])
                    plt.legend()
                plt.xlabel('Data point of measurement period')
                plt.show()

    def plot_RE(self):
        for i in self.columns:
            fig, ax = plt.subplots(len(self.models), 1, figsize=self.figsize)
            for idx, model in enumerate(self.models):
                _ax = ax[idx] if len(self.models) > 1 else ax
                _ax.set_ylabel(model.capitalize())

                mse_period = self.mse_period[model][i].to_numpy().tolist()
                anom_start = self.anomalies_period[model][i]
                threshold = self.threshold[model][i]
                ylim = 5 * threshold

                _ax.plot(mse_period, label='RE')
                _ax.axhline(y=threshold, color='r', linestyle='-', label='Threshold')
                _ax.plot(anom_start, mse_period[anom_start], 'ro', label='Anomaly prediction')
                _ax.set_ylim(-0.02*ylim, ylim)
                # _ax.set_xticks([0, len(mse_period)], ["start", "end"])
                _ax.set_yticks([0, ylim], ["0", f"{np.format_float_scientific(ylim, precision = 1, exp_digits=1)}"])
                # plot dot where anomaly starts

            # plot a vertical text left on the figure to describe the y-axis
            handles, labels = _ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower right')
            fig.supxlabel('Duration of experiment')
            fig.supylabel('Reconstruction Error')
            fig.suptitle(f'Reconstruction Error and Anomaly Threshold', fontsize=14)
            fig.show()

            return [mse_period, anom_start, threshold]

    def plot_certainty(self, center_factor=.5, max_factor=2.):
        """ Plot a figure that shows a color bar for the certainty of the anomaly """

        for i in self.columns:
            fig, ax = plt.subplots(len(self.models), 1, figsize=self.figsize)
            for idx, model in enumerate(self.models):
                _ax = ax[idx] if len(self.models) > 1 else ax
                _ax.set_ylabel(model.capitalize())

                mse = self.mse_period[model][i].to_numpy()
                mean = self.mse_val[model][i].mean()
                diff = self.threshold[model][i] - mean
                is_anomal_bearing = i in anomal_bearings[self.experiment_name]

                # plot the color bar
                divnorm = colors.TwoSlopeNorm(vmin=mean, vcenter=mean + center_factor * diff, vmax=mean + max_factor * diff)
                im = _ax.imshow(mse.reshape(1, -1), cmap='bwr', aspect='auto', norm=divnorm, alpha=.8)
                # add vertical lines to indicate the anomaly start and end
                _ax.axvline(x=self.anomalies_period[model][i], color='red', linewidth=3, linestyle='--', label='Anomaly start')
                _ax.set_yticks([])
            # plot color bar legend with custom ticks
            handles, labels = _ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower left')
            cb = fig.colorbar(im, ax=ax, ticks=[mean, mean + center_factor * diff, mean + diff, mean + max_factor * diff])
            cb.ax.set_yticklabels(['mean RE', f'{int(center_factor*100)}% Threshold', '100% Threshold', f'{int(max_factor*100)}% Threshold'])
            fig.supxlabel('Measurement Period')
            fig.suptitle(f'Certainty of Defect', fontsize=14)
            fig.show()

    def plot_certainty_transfer(self, center_factor=.5, max_factor=2.):
        for i in self.columns:
            fig, ax = plt.subplots(2, 1, figsize=self.figsize)
            title = ["Directly trained", "Transfer learning"]
            for idx, model in enumerate(['federated', 'federated_transfer']):
                _ax = ax[idx]
                _ax.set_ylabel(title[idx])

                mse = self.mse_period[model][i].to_numpy()
                mean = self.mse_val[model][i].mean()
                diff = self.threshold[model][i] - mean
                is_anomal_bearing = i in anomal_bearings[self.experiment_name]

                # plot the color bar
                divnorm = colors.TwoSlopeNorm(vmin=mean, vcenter=mean + center_factor * diff,
                                              vmax=mean + max_factor * diff)
                im = _ax.imshow(mse.reshape(1, -1), cmap='bwr', aspect='auto', norm=divnorm, alpha=.8)
                # add vertical lines to indicate the anomaly start and end
                _ax.axvline(x=self.anomalies_period[model][i], color='red', linewidth=3, linestyle='--',
                            label='Anomaly start')
                _ax.set_yticks([])
            # plot color bar legend with custom ticks
            handles, labels = _ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower left')
            cb = fig.colorbar(im, ax=ax,
                              ticks=[mean, mean + center_factor * diff, mean + diff, mean + max_factor * diff])
            cb.ax.set_yticklabels(['mean RE', f'{int(center_factor * 100)}% Threshold', '100% Threshold',
                                   f'{int(max_factor * 100)}% Threshold'])
            fig.supxlabel('Measurement Period')
            fig.suptitle(f'Certainty of Defect', fontsize=14)
            fig.show()

    def plot_cumulate_certainty(self, ylim=100):

        for i in self.columns:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            _ax = ax
            for model in self.models:
                mse = self.mse_period[model][i].to_numpy()[self.rolling_min-1:]
                threshold = self.threshold[model][i]
                mse_over_threshold = mse - threshold
                mse_over_threshold[mse_over_threshold < 0] = 0
                mse_over_threshold = mse_over_threshold / threshold
                cummulative_mse = mse_over_threshold.cumsum()
                cummulative_mse = cummulative_mse / 4 if model != 'baseline' else cummulative_mse
                # color the area under the curve
                _ax.plot(cummulative_mse, label=model.capitalize())
                # _ax.fill_between(range(len(mse)), 0, cummulative_mse, color='red', alpha=0.5)
            _ax.set_xticks([0, int(0.2*len(mse)), int(0.4*len(mse)), int(0.6*len(mse)), int(0.8*len(mse)), len(mse)],
                           ["start", "20%", "40%", "60%", "80%", "end"])
            _ax.set_ylim(0, ylim)


            fig.supxlabel('Duration of experiment')
            fig.suptitle(f'Cummulative Certainty of Defect, with rolling window of size {self.rolling_min} '
                         f'\n{self.experiment_name}', fontsize=14)
            fig.show()

    def plot_resources(self):
        y_labels = ["Duration [s]", "Duration [s]", "Utilization [%]", "Memory Allocation [GB]"]
        titels = ["Training Runtime", "Time / Epoch at 25% CPU", "CPU Utilization", "Memory Allocation"]
        fig, ax = plt.subplots(2, 2, figsize=(10, 8))
        # set the space between the subplots
        fig.subplots_adjust(hspace=0.28, wspace=0.28)
        for idx, resource in enumerate(self.resource_results):
            _ax = ax[idx // 2, idx % 2]
            models = ['Federated', 'Centralized', 'Baseline']
            width = 0.35
            if idx == 0:
                resource_values = [self.resource_results[resource][i] for i in range(3)]
                _ax.bar(range(len(models)), resource_values, width * 2)
                # stack an additional bar on the first one
                resource_values2 = resource_values.copy()
                resource_values2[0] = resource_values2[0] * 3
                resource_values2[1] = 0
                resource_values2[2] = 0
                _ax.bar(range(len(models)), resource_values2, width * 2, bottom=resource_values)
            elif idx == 1 or idx == 2:
                resource_values = [self.resource_results[resource][i] for i in range(3)]
                _ax.bar(range(len(models)), resource_values, width*2)
            else:
                memory_mean, memory_max = mem_resources[0], mem_resources[1]
                x = np.arange(len(models))
                _ax.bar(x - width/2, memory_mean, width, label='Avg. Memory')
                _ax.bar(x + width/2, memory_max, width, label='Max. Memory', color='darkred')
                _ax.legend()
            _ax.set_xticks(range(len(models)), models)
            _ax.set_ylabel(y_labels[idx])
            _ax.set_title(titels[idx])
        fig.supxlabel('Model type')
        fig.supylabel('Resource usage')
        fig.suptitle(f"Comparison of resource demands of different models")
        fig.show()

    @staticmethod
    def plot_all_experiments_federated():

        # load experiments
        experiments = {'1': None, '2': None, '3': None}
        for i in experiments:
            name = f"bearing_experiment-{i}"
            with open(f"logs/{name}/federated.json", 'r') as f:
                experiments[i] = json.load(f)
            experiments[i] = np.array([experiments[i][str(j)]['lstm']['mse'] for j in range(4)]).T
            experiments[i] = pd.DataFrame(experiments[i])
        faulty_bearings = [experiments['1'][2], experiments['1'][3], experiments['2'][0], experiments['3'][2]]
        # standardize the faulty_bearings
        scaler = StandardScaler()
        # faulty_bearings = [scaler.fit_transform(np.array(faulty_bearings[i]).reshape(-1, 1)) for i in range(4)]
        plt.ylim(0, 30)
        keydict = {'linewidth': 1, 'alpha': 0.9}
        label = ['E1 - Bearing 3', 'E1 - Bearing 4', 'E2 - Bearing 1', 'E3 - Bearing 2']
        fig, ax = plt.subplots(2, 2, figsize=(10, 8))
        for i in range(4):
            _ax = ax[i // 2, i % 2]
            _ax.plot(faulty_bearings[i], label=label[i], **keydict)
            _ax.set_title(label[i])
            _ax.set_ylim(0, 10)
        fig.show()

    def run(self):
        self.plot_data(ylim=7)
        self.plot_RE()
        self.plot_certainty(center_factor=.5, max_factor=2.)
        self.plot_period(chunk_numbers=[970], mean_window=3, line_width=0.5)
        self.plot_resources()
        if 'federated_transfer' in self.models:
            self.plot_certainty_transfer()


def combine_federated_json(experiment_name="bearing_experiment-2"):
    results = {"0": {}, "1": {}, "2": {}, "3": {}}
    for i in [0, 1, 2, 3]:
        results[str(i)] = json.load(open(f"logs/{experiment_name}/federated_CLIENT_{i}.json"))
    # save the results
    with open(f"logs/{experiment_name}/federated.json", "w") as f:
        json.dump(results, f)


def plot_all_experiments():
    results = [0] * 4
    results[0] = Plotter(columns=[2], rolling_min=3, experiment_name='bearing_experiment-1',
                         models=["federated"]).plot_RE()
    results[1] = Plotter(columns=[3], rolling_min=3, experiment_name='bearing_experiment-1',
                         models=["federated"]).plot_RE()
    results[2] = Plotter(columns=[0], rolling_min=3, experiment_name='bearing_experiment-2',
                         models=["federated"]).plot_RE()
    results[3] = Plotter(columns=[2], rolling_min=3, experiment_name='bearing_experiment-3',
                         models=["federated"]).plot_RE()

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    titles = ["E1 - Bearing 3", "E1 - Bearing 4", "E2 - Bearing 1", "E3 - Bearing 2"]
    for i in range(4):
        re, anom_start, threshold = results[i]
        ylim = 5 * threshold
        _ax = ax[i // 2, i % 2]
        _ax.plot(re, label="RE")
        _ax.axhline(y=threshold, color='r', linestyle='-', label='Threshold')
        _ax.plot(anom_start, re[anom_start], 'ro', label='Anomaly prediction')
        _ax.set_title(titles[i])
        _ax.set_ylim(0, ylim)
        _ax.set_yticks([0, ylim], ["0", f"{np.format_float_scientific(ylim, precision=1, exp_digits=1)}"])
    handles, labels = _ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower left')
    fig.supxlabel('Measurement Period of experiment')
    fig.supylabel('Reconstruction Error')
    fig.show()


if __name__ == '__main__':

    # plot experiment 2 for faulty (0) and healty (1) bearing
    experiment_name = f"bearing_experiment-2"
    bearings = [0, 1]

    plotter = Plotter(columns=bearings, rolling_min=3,
                      experiment_name=experiment_name,
                      models=["federated", "centralized", "baseline"]
                      )
    plotter.run()

