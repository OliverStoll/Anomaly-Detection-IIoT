import os

from matplotlib import pyplot as plt
import pandas as pd
from datetime import datetime
import psutil


def plot_tuning_logs(path="hyper_tuning/tuning_log.txt"):

    full_text = open(path, "r").read()
    trial_list = full_text.split("Search: Running Trial")
    all_hyperpars = []
    all_val_losses = []
    for trial in trial_list[1:]:
        try:
            hyperparameters = trial.split("HYPERPARAMETERS:")[1].split('\n')[0].split(';')
            hyperparameters[1] = f"{float(hyperparameters[1]):.0e}"  # display learning rate as e
            val_loss = trial.split("\nval_loss: ")[1].split('\n')[0]
            all_hyperpars.append(hyperparameters)
            all_val_losses.append(float(val_loss))
        except:
            pass

    # get strings from hyperparameters
    all_labels = []
    for hyperpar in all_hyperpars:
        label = ""
        for i in range(len(hyperpar)):
            label += hyperpar[i] + " "
        all_labels.append(label)

    print(all_labels)

    # sort by val_loss
    all_val_losses, all_labels = zip(*sorted(zip(all_val_losses, all_labels), reverse=True))

    # Plot the figure.
    freq_series = pd.Series(all_val_losses)
    plt.figure(figsize=(12, 12))
    ax = freq_series.plot(kind="barh")
    ax.set_xscale('log')
    ax.set_title("Hyperparameter Tuning")
    ax.set_xlabel("Validation Loss")
    ax.set_yticklabels(all_labels)

    plt.show()


def log_ressource_usage(path="logs/ressources"):
    pid = psutil.Process().pid

    current_path = os.path.realpath(__file__)
    parent_path = os.path.dirname(current_path)
    grand_parent_path = os.path.dirname(parent_path)
    log_path = f'{grand_parent_path}/{path}.txt'
    # print full path of current file
    print()
    with open(log_path, 'w') as f:  # clear file
        pass

    while True:
        cpu_usage = psutil.Process(pid).cpu_percent(interval=1)
        # print("\nCPU:", cpu_usage)

        # get total memory usage of process
        memory_total = psutil.Process(pid).memory_info().rss / (1024 * 1024)

        with open(log_path, 'a') as f:
            f.write(f"{datetime.now()},{cpu_usage},{memory_total}\n")  # write to file


def get_ressource_usage(path="logs/ressources.txt"):

    ressource_usages = []
    full_text = open(path, "r").read()  # read full text

    # split by experiment
    experiment_list = full_text.split("START ")
    if "START centralized" in full_text:
        experiment_list = experiment_list[1:]

    # hacky way to replace empty federated with federated: client_0 ressources
    client_0_path = path.replace('ressources', 'ressources_CLIENT_0')
    federated_full_text = open(client_0_path, "r").read()  # read full text

    experiment_list[1] = federated_full_text

    # get all cpu usage
    for experiment in experiment_list:
        ressource_usage_dict = {}
        lines = experiment.split("\n")[1:-2]
        cpu_usages = [float(line.split(",")[1]) for line in lines]
        mem_usages = [float(line.split(",")[2]) for line in lines]
        ressource_usage_dict["cpu"] = round(sum(cpu_usages), 2)
        ressource_usage_dict["memory"] = round(sum(mem_usages), 2)
        ressource_usage_dict["memory_mean"] = round(sum(mem_usages) / len(mem_usages), 2)
        ressource_usage_dict["memory_max"] = round(max(mem_usages), 2)
        ressource_usages.append(ressource_usage_dict)




    return {'centralized': ressource_usages[0], 'federated': ressource_usages[1], 'baseline': ressource_usages[2]}


class TFLogReader:
    """ Class that reads the logs from the tf experiments and returns a dictionary of metrics """
    def __init__(self, path):
        self.path = path
        self.log_type = self.get_log_type()
        self.lstm_rounds, self.fft_rounds = self.get_rounds_from_tf_log()
        self.metrics = {'lstm': self.get_metrics_from_rounds(self.lstm_rounds),
                        'fft': self.get_metrics_from_rounds(self.fft_rounds)}

    def get_log_type(self):
        """ Returns the type of log (centralized, federated, baseline) """
        if "centralized" in self.path:
            return "centralized"
        elif "federated" in self.path:
            return "federated"
        else:
            return "baseline"

    def get_rounds_from_tf_log(self):
        """ Function that extracts the lines for lstm and fft for the correct experiment type """

        lstm_rounds = []
        fft_rounds = None

        # read the file
        full_text = open(self.path, "r").read()

        # extract the relevant lines from the logs and sort them by model

        if self.log_type == 'centralized':
            all_rounds = full_text.split("Epoch ")[1:]
            lstm_rounds = all_rounds[:len(all_rounds) // 2]
            fft_rounds = all_rounds[len(all_rounds) // 2:]
        elif self.log_type == 'federated':
            all_rounds = full_text.split("Round ")[1:]
            lstm_rounds = [round.split("\n")[1] for round in all_rounds]
            fft_rounds = [round.split("\n")[2] for round in all_rounds]
        else:
            pass  # TODO: baseline logs

        return lstm_rounds, fft_rounds

    @staticmethod
    def get_metrics_from_rounds(rounds):
        """ Returns a dictionary of metrics for the given rounds (lstm or fft) """

        if rounds is not None:
            round_metrics = {
                'losses': [float(round.split('loss: ')[1].split(' -')[0]) for round in rounds],
                'val_losses': [float(round.split('val_loss: ')[1].split(' - ')[0]) for round in rounds],
                'time_per_step': [float(round.split('/step')[0].split(' - ')[-1].replace('ms', '').replace('s', '000')) for round
                                  in rounds]
            }
        else:
            round_metrics = {}
        return round_metrics

    def plot_metrics(self):
        """ Plots the metrics for the given log """

        for metric_name in self.metrics["lstm"].keys():
            plt.plot(self.metrics["lstm"][metric_name], label=f"LSTM {metric_name}")
            plt.plot(self.metrics["fft"][metric_name], label=f"FFT {metric_name}")
            plt.legend()
            plt.title(f"{metric_name}")
            plt.show()


if __name__ == "__main__":

    # log_ressource_usage()
    # metrics = TFLogReader("logs/centralized/bearing_experiment-2/bearing-1.txt").metrics['lstm']['time_per_step']
    get_ressource_usage()