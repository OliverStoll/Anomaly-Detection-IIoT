import os
import tensorflow as tf

from training import Training
from util.config import c, client_config


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
    # TODO: run worker_aggregation and multiple worker_training with their 'CLIENT_NAME' environment variable
    pass


def evaluate_baseline_model(model_path):
    """ Evaluate the baseline model """
    # TODO: get the mse from the model
    os.chdir('./baseline')  # change inside directory of baseline
    from baseline.evaluate_baseline import plot_and_evaluate
    lstm_baseline = tf.keras.models.load_model(model_path)
    mse = plot_and_evaluate(lstm_autoencoder=lstm_baseline)
    print(f"Baseline model MSE: {mse}")
    print(mse.shape)
    os.chdir('../')  # change back to parent directory


def evaluate_single_model(model_path):
    dataset_path, data_columns = c.CLIENT_1['DATASET_PATH'], c.CLIENT_1['DATASET_COLUMNS']
    trainer = Training(data_path=dataset_path, data_columns=data_columns)
    trainer.load_models(dir_path=model_path)
    trainer.evaluate(show_as=True, show_preds=True, show_roc=True)


def evaluate_models():
    """ Evaluate a model based on the given path """

    # evaluate centralized model
    os.environ['CLIENT_NAME'] = 'CENTRALIZED'
    evaluate_single_model(model_path=f"model/centralized/{client_config['MODEL_PATH']}")

    # evaluate federated model
    os.environ['CLIENT_NAME'] = 'FEDERATED'
    evaluate_single_model(model_path=f"model/federated/{client_config['MODEL_PATH']}")

    # evaluate baseline model
    os.environ['CLIENT_NAME'] = 'BASELINE'
    evaluate_baseline_model(model_path=f"model/baseline/{client_config['MODEL_PATH']}")


if __name__ == '__main__':
    train_centralized()
    train_baseline()
    evaluate_models()
