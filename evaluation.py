import os
import sys

import tensorflow as tf
import subprocess as sp
import threading

from training import Training
from util.config import c, config, client_config


def train_baseline():
    """ Run the baseline experiment to train the baseline model """
    os.chdir('./baseline')  # change inside directory of baseline
    import baseline.baseline
    os.chdir('../')  # change back to parent directory


def train_centralized():
    """ Train the centralized model using the Trainer class """
    trainer = Training(data_path=client_config['DATASET_PATH'], data_columns=client_config['DATASET_COLUMNS'])
    trainer.train_models(epochs=c.EPOCHS)
    trainer.save_models(dir_path=f"model/centralized/{client_config['MODEL_PATH']}")


def train_federated():
    """ Train the federated model using the worker classes and threading """

    def run_file(file_path, client_name, stdout,
                 activate_path="C:/DRIVE/SOFTWARE/_Archiv/_global-venv-pc1/Scripts/activate.bat"):
        """ Start a subprocess to run either worker aggregation or worker training """
        os.environ['CLIENT_NAME'] = client_name
        sp.run(f"{activate_path} && python {file_path}", shell=True, stdout=stdout)

    # create thread with run file function and start it
    t_aggregation = threading.Thread(target=run_file, args=(f"worker_aggregation.py", "AGGREGATION", None))
    t_aggregation.start()
    t_worker_0 = threading.Thread(target=run_file, args=(f"worker_training.py", "CLIENT_0", None))
    t_worker_0.start()

    for worker_id in range(1, c.NUM_CLIENTS):
        # create a stdout file for the subprocesses
        logs_path = f"logs/{client_config['DATASET_PATH'].replace('data/', '').replace('bearing/', 'bearing_')}"
        os.makedirs(logs_path, exist_ok=True)
        logs_file = open(f"{logs_path}/{worker_id}.txt", "w")
        t_worker = threading.Thread(target=run_file, args=(f"worker_training.py", f"CLIENT_{worker_id}", logs_file))
        t_worker.start()

    # wait for all threads to finish
    t_aggregation.join()


def evaluate_baseline_model(model_path):
    """ Evaluate the baseline model """
    # TODO: get the mse from the model
    lstm_baseline = tf.keras.models.load_model(model_path)
    os.chdir('./baseline')  # change inside directory of baseline
    from baseline.evaluate_baseline import plot_and_evaluate
    mse = plot_and_evaluate(lstm_autoencoder=lstm_baseline)
    os.chdir('../')  # change back to parent directory4
    print(f"Baseline model MSE: {mse}")
    print(mse.shape)


def evaluate_single_model(model_path, show, client_config):
    dataset_path, data_columns = client_config['DATASET_PATH'], client_config['DATASET_COLUMNS']
    trainer = Training(data_path=dataset_path, data_columns=data_columns)
    trainer.load_models(dir_path=model_path)
    trainer.evaluate(show_as=('show_as' in show), show_preds=('show_pred' in show), show_roc=('show_roc' in show))


def train_models(model_types):
    if 'baseline' in model_types:
        train_baseline()
    if 'centralized' in model_types:
        train_centralized()
    if 'federated' in model_types:
        train_federated()


def evaluate_models(show, evaluate_list):
    """ Evaluate a model based on the given path """

    # evaluate baseline model
    if 'baseline' in evaluate_list:
        os.environ['CLIENT_NAME'] = 'BASELINE'
        evaluate_baseline_model(model_path=f"model/baseline/{client_config['MODEL_PATH']}/lstm_baseline.h5")

    # evaluate centralized model
    if 'centralized' in evaluate_list:
        os.environ['CLIENT_NAME'] = 'CENTRALIZED'
        evaluate_single_model(model_path=f"model/centralized/{client_config['MODEL_PATH']}", show=show,
                              client_config=c.CLIENT_0)

    # evaluate federated model
    if 'federated' in evaluate_list:
        # iterate over all clients
        client_names = [name for name in config.keys() if 'CLIENT_' in name]
        for name in client_names:
            os.environ['CLIENT_NAME'] = name
            evaluate_single_model(model_path=f"model/federated/{config[name]['MODEL_PATH']}", show=show,
                                  client_config=config[name])


if __name__ == '__main__':

    train = ['federated']
    evaluate = ['federated']
    show = ['show_as', 'show_pred', 'show_roc']

    train_models(model_types=train)
    evaluate_models(evaluate_list=evaluate, show=show)
