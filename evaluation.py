import os
import sys

import tensorflow as tf
import subprocess as sp
import threading
import time
import numpy as np
import yaml
import json
from datetime import datetime

from training import Training
from plotting import MiscPlotter, PredictionsPlotter
from util.ml_calculations import calculate_f1, get_timetuples_from_indexes, get_anomaly_indexes
from util.config import c, config, client_config, baseline_config
from util.logs import log_ressource_usage, get_ressource_usage

experiment_path = client_config['MODEL_PATH'].split('/')[0]
anomalies = yaml.safe_load(open(f"configs/anomalies.yaml"))

# set tf error logging to none
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class TrainingEvaluator:
    def __init__(self, train_only_cpu, measure_ressources=True, log_path=c.LOGS_PATH, model_types=None):
        self.log_path = log_path
        self.log_ressources_path = f"{log_path}/ressources.txt"
        self.model_types = model_types if model_types is not None else ['baseline', 'centralized', 'federated']
        self.old_stdout = sys.stdout
        self.times = {}
        self.resources = {}

        if train_only_cpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable gpu usage
            config_tf = tf.compat.v1.ConfigProto()  # set max threads for tensorflow
            config_tf.intra_op_parallelism_threads = 4
            config_tf.inter_op_parallelism_threads = 4

        # start thread for measuring ressources
        if measure_ressources:
            self.t_ressources = threading.Thread(target=log_ressource_usage,
                                                 args=[f"{log_path}/ressources"]
                                                 ).start()
            time.sleep(1)

    def train_models(self):
        """ Train the models and measure runtime and memory/cpu usage """

        times = {'baseline': {}, 'centralized': {}, 'federated': {}}

        for model_type in self.model_types:
            time.sleep(1)

            # TODO: start logging in each function!!!!!!
            # log start time
            with open(f"{self.log_path}/ressources.txt", "a") as f:
                f.write(f"\nSTART {model_type}\n")
            times[model_type]['start'] = datetime.now()

            # train model

            if model_type == 'centralized':
                self.train_centralized()
            elif model_type == 'federated':
                self.train_federated()
            elif model_type == 'baseline':
                self.train_baseline()

            # log end time
            times[model_type]['runtime'] = round((datetime.now() - times[model_type]['start']).total_seconds(), 2)
            times[model_type]['end'] = datetime.now().strftime("%H:%M:%S.%f")
            times[model_type]['start'] = times[model_type]['start'].strftime("%H:%M:%S.%f")

        if self.t_ressources is not None:
            self.t_ressources.join()

        self.times = times
        for model_type in self.model_types:
            self._adjust_runtime(model_type)
        self.resources = get_ressource_usage(f"{self.log_path}/ressources.txt")
        for model_type in self.model_types:
            self.resources[model_type]['runtime'] = round(self.times[model_type]['runtime'], 3)
            self.resources[model_type]['cpu'] = self.resources[model_type]['cpu'] / 100  # convert to core-secs

        print("Training finished\n")
        self.print_results()

        # save results to json file
        with open(f"{self.log_path}/resources.json", "w") as f:
            json.dump(self.resources, f, indent=4)

        return self.resources

    def train_baseline(self):
        """ Run the baseline experiment to train the baseline model """

        print("Training baseline model")

        sys.stdout = self._create_log("baseline", client_config['DATASET_COLUMNS'])
        os.chdir('./baseline')  # change inside directory of baseline
        import baseline.baseline
        os.chdir('../')  # change back to parent directory
        sys.stdout = self.old_stdout

    def train_centralized(self):
        """ Train the centralized model using the Trainer class """

        print("Training centralized model")
        sys.stdout = self._create_log("centralized", client_config['DATASET_COLUMNS'])
        start = datetime.now()
        trainer = Training(data_path=client_config['DATASET_PATH'], data_columns=client_config['DATASET_COLUMNS'])
        trainer.verbose = 2
        time_until_training = (datetime.now() - start).total_seconds()
        print(f"TIME_UNTIL_TRAINING:{time_until_training:.2f}")
        trainer.train_models(epochs=c.EPOCHS)
        trainer.save_models(dir_path=f"model/{config['EXPERIMENT_NAME']}/centralized")
        sys.stdout = self.old_stdout

    def train_federated(self):
        """ Train the federated model using the worker classes and threading """

        def run_file(file_path, client_name, stdout,
                     activate_path="C:/DRIVE/SOFTWARE/_Archiv/_global-venv-pc1/Scripts/activate.bat"):
            """ Start a subprocess to run either worker aggregation or worker training """
            os.environ['CLIENT_NAME'] = client_name
            sp.run(f"{activate_path} && python {file_path}", shell=True, stdout=stdout)

        print("Training federated model")

        # create thread with run file function and start it
        t_aggregation = threading.Thread(target=run_file, args=(f"worker_aggregation.py", "AGGREGATION", None))
        t_aggregation.start()

        for worker_id in range(c.NUM_CLIENTS):
            log_file = self._create_log("federated", str(worker_id))  # create a log file for the subprocesses
            t_worker = threading.Thread(target=run_file, args=(f"worker_training.py", f"CLIENT_{worker_id}", log_file))
            t_worker.start()

        # wait for all threads to finish
        t_aggregation.join()

    def print_results(self):
        """ Get the metrics of the training process (runtime and memory/cpu usage) """

        for model in self.model_types:
            result_str = f"{model}:\n" \
                         f"RUNTIME[s]: {self.times[model]['runtime']:.1f}" \
                         f" - CPU[core-secs]: {self.resources[model]['cpu'] / 100:.1f}" \
                         f" - CPU[%]: {self.resources[model]['cpu'] / self.times[model]['runtime']:.1f}" \
                         f" - MEMORY_MEAN[MB]: {self.resources[model]['memory_mean']:.1f}" \
                         f" - MEMORY_MAX[MB]: {self.resources[model]['memory_max']:.1f}\n"
            print(result_str)

    @staticmethod
    def _create_log(train_type, log_name):
        log_path = f"logs/{experiment_path}/{train_type}"
        os.makedirs(log_path, exist_ok=True)
        log_file = open(f"{log_path}/{log_name}.txt", "w")
        return log_file

    def _adjust_runtime(self, train_type):
        path = f"{c.LOGS_PATH}/{train_type}"
        # get first file in path directory
        first_file_path = os.listdir(path)[0]
        with open(f"{path}/{first_file_path}", "r") as file:
            time_until_train = float(file.read().split("TIME_UNTIL_TRAINING:")[1].split("\n")[0])
            self.times[train_type]['runtime'] -= time_until_train


class ModelEvaluator:
    """ Evaluate all specified models and determine the f1 scores of each model """
    def __init__(self, evaluation_plots, types_to_evaluate):
        self.plot_list = evaluation_plots
        self.evaluate_list = types_to_evaluate
        self.config_client_names = [name for name in config.keys() if 'CLIENT_' in name]
        self.models_dir = f"model/{config['EXPERIMENT_NAME']}"

    def evaluate_models(self):
        """ Evaluate all models, and return (& save) the f1 scores. """

        f1_scores = {}  # federated got client_name as key

        # evaluate baseline model
        if 'baseline' in self.evaluate_list:
            print("BASELINE")
            os.environ['CLIENT_NAME'] = 'BASELINE'
            f1_scores['baseline'] = self.evaluate_baseline_model(model_path=f"{self.models_dir}/baseline/lstm.h5",
                                                                 client_config=config['CLIENT_0'])

        # evaluate centralized model
        if 'centralized' in self.evaluate_list:
            print("CENTRALIZED")
            os.environ['CLIENT_NAME'] = 'CENTRALIZED'
            f1_scores['centralized'] = self.evaluate_single_model(model_path=f"{self.models_dir}/centralized",
                                                                  client_config=config['CLIENT_0'])

        # evaluate federated model
        if 'federated' in self.evaluate_list:
            print("FEDERATED")
            f1_scores['federated'] = {}
            for name in self.config_client_names[:config['NUM_CLIENTS']]:  # iterate over all clients
                os.environ['CLIENT_NAME'] = name
                f1_scores["federated"][name] = self.evaluate_single_model(
                    model_path=f"{self.models_dir}/federated/{name}",
                    client_config=config[name])

        # save f1 scores as json to log path
        with open(f"{c.LOGS_PATH}/f1_scores.json", "w") as file:
            json.dump(f1_scores, file, indent=4)

        return f1_scores

    def evaluate_single_model(self, model_path, client_config):
        """ Evaluate a single model (centralized or federated) """

        show = self.plot_list
        dataset_path, data_columns = client_config['DATASET_PATH'], client_config['DATASET_COLUMNS']
        trainer = Training(data_path=dataset_path, data_columns=data_columns)
        trainer.load_models(dir_path=model_path)
        trainer.evaluate(show_as=('show_as' in show), show_preds=('show_pred' in show), show_roc=('show_roc' in show))
        return trainer.f1s

    def evaluate_baseline_model(self, model_path, client_config):
        """ Evaluate the baseline model """
        lstm_baseline = tf.keras.models.load_model(model_path)
        os.chdir('./baseline')  # change inside directory of baseline
        from baseline.evaluate_baseline import plot_and_evaluate
        mse = plot_and_evaluate(lstm_autoencoder=lstm_baseline)
        os.chdir('../')  # change back to parent directory4

        # get the threshold
        start_index = int(c.THRESHOLD_CALCULATION_PERIOD[0] * len(mse))
        end_index = int(c.THRESHOLD_CALCULATION_PERIOD[1] * len(mse))
        mse_period = np.array(mse[start_index:end_index])
        threshold = np.mean(mse_period) + np.std(mse_period) * c.THRESHOLD_DEVIATIONS

        # get the labels
        dataset_path_split = client_config['DATASET_PATH'].split('/')
        labels = anomalies[dataset_path_split[1]][dataset_path_split[2]][client_config['MODEL_PATH'].split('/')[-1]]

        # get the f1 score
        f1 = calculate_f1(mse=mse, threshold=threshold, labels=labels, is_until_failure=True)
        print(f"lstm F1:{f1: .3f}")

        # plot
        if 'show_as' in self.plot_list:
            miscplotter = MiscPlotter(trainer=None)
            miscplotter.plot_anomaly_scores(thresholds={'lstm': threshold}, mses={'lstm': mse})

        if 'show_pred' in self.plot_list:
            features = len(client_config['DATASET_COLUMNS'])
            bearing_index = int(client_config['DATASET_COLUMNS'][0] / features)
            anomaly_indexes = get_anomaly_indexes(mse=mse, threshold=threshold,
                                                  is_until_failure=('bearing' in client_config['DATASET_PATH']))
            anomaly_timetuples = get_timetuples_from_indexes(indexes=anomaly_indexes, max_index=len(mse))
            PredictionsPlotter(file_path=f"{client_config['DATASET_PATH']}_{c.SPLIT}.csv",
                               sub_experiment_index=bearing_index,
                               ylim=c.PLOT_YLIM_VIBRATION,
                               features=features,
                               anomalies_real=anomalies[dataset_path_split[1]][dataset_path_split[2]],
                               anomalies_pred_lstm=anomaly_timetuples,
                               ).plot_experiment()

        return {'lstm': round(f1, 4)}


def train_and_evaluate_experiments(train_types, eval_types, plot_types, train_flag):

    results_path = f"{c.LOGS_PATH}/results.json"
    data_usage_path = f"{c.LOGS_PATH}/data_usage.json"

    # load old results from json file
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
    else:
        results = {}

    results_metadata = {'metadata': {'split': c.SPLIT,
                                     'baseline_split': baseline_config['DOWNSAMPLE'],
                                     'experiment_name': config['EXPERIMENT_NAME'],
                                     'experiment_columns': str(client_config['DATASET_COLUMNS'])}}
    results.update(results_metadata)

    # train models and update results with ressource usage
    if train_flag:
        train_results = TrainingEvaluator(model_types=train_types, train_only_cpu=True).train_models()
        with open(data_usage_path, 'r') as f:
            results['data_usage'] = json.load(f)
        results.update(train_results)

    # evaluate and recalculate results with f1 scores
    f1_scores = ModelEvaluator(types_to_evaluate=eval_types, evaluation_plots=plot_types).evaluate_models()
    for key in f1_scores.keys():
        if key in results:
            results[key]['f1'] = f1_scores[key]
        else:
            results[key] = {'f1': f1_scores[key]}

    # save results
    with open(f"{config['LOGS_PATH']}/results.json", 'w') as f:
        json.dump(results, f, indent=4)

    return results


if __name__ == '__main__':
    train = ['centralized', 'federated', 'baseline']
    evaluate = ['centralized', 'federated', 'baseline']
    plots = ['show_as']
    train_flag = True

    all_results = train_and_evaluate_experiments(train_types=train,
                                                 eval_types=evaluate,
                                                 plot_types=plots,
                                                 train_flag=train_flag)
