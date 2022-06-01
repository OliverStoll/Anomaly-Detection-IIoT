import socket
import pickle
import os
import random
from keras.models import load_model

from training import Training
from util.socket_functionality import send_msg, recv_msg
from util.config import c, client_config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class TrainingWorker:
    """
    This worker class is responsible for training the model.

    It encapsulates the training process.
    Additionally, it handles the communication with the aggregation worker and runs the full
    training cycle.
    """

    def __init__(self, connect_ip_port: (str, int), data_path: str, data_cols: list, model_path: str):
        self.trainer = Training(data_path=data_path, data_columns=data_cols)
        self.model_path = model_path
        self.epoch = 0
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((connect_ip_port[0], connect_ip_port[1]))
        print(f"TRAININGWORKER: Connected to {connect_ip_port}")

    def train_round(self, epochs):
        self.trainer.train_models(epochs=epochs)  # TODO: check efficiency
        self.epoch += epochs

    def send_weights(self):
        weights = self.trainer.model_fft.get_weights()
        weights_data = pickle.dumps(weights)
        send_msg(sock=self.socket, msg=weights_data)

    def receive_weights(self):
        weights_data = recv_msg(sock=self.socket)
        weights = pickle.loads(weights_data)
        self.trainer.model_fft.set_weights(weights)

    def run(self, rounds, epochs_per_round):
        for i in range(rounds):
            print(f"Round {i}")
            self.train_round(epochs=epochs_per_round)
            self.send_weights()
            self.receive_weights()
        self.trainer.save_models(self.model_path)
        self.trainer.evaluate(show_all=True)


if __name__ == '__main__':
    # print all environment variables
    print(f"TRAINING_WORKER: Starting as {os.environ.get('CLIENT_NAME')} ")
    trainer = TrainingWorker(connect_ip_port=c.CONNECT_IP_PORT,
                             data_path=client_config['DATASET_PATH'],
                             data_cols=client_config['DATASET_COLUMNS'],
                             model_path=f"model/federated/{client_config['MODEL_PATH']}")
    trainer.run(rounds=c.EPOCHS, epochs_per_round=1)

