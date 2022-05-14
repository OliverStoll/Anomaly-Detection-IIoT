import socket
import os
from keras.models import load_model

from training import Training
from evaluation import evaluate_model_lstm, evaluate_model_fft
from functionality.socket_functions import send_msg, recv_msg
from functionality.config import c, client_c

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class TrainingWorker:
    """
    This worker class is responsible for training the model.

    It encapsulates the training process.
    Additionally, it handles the communication with the aggregation worker and runs the full
    training cycle.
    """

    def __init__(self, server_ip_port, data_path, data_cols, model_path):
        self.trainer = Training(data_path=data_path, data_columns=data_cols)
        self.model_path = model_path
        self.epoch = 0
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect(server_ip_port)
        print(f"TRAININGWORKER: Connected to {server_ip_port}")

    def train_round(self, epochs):
        self.trainer.train_models(epochs=epochs)  # TODO: check efficiency
        self.epoch += epochs

    def send_weights(self):
        self.trainer.model_fft.save(f"{self.model_path}/model_fft.h5")
        with open(f"{self.model_path}/model_fft.h5", "rb") as file:
            data = file.read()
        send_msg(sock=self.socket, msg=data)

    def receive_weights(self):
        with open(f"{self.model_path}/model_fft_re.h5", "wb") as file:
            data = recv_msg(sock=self.socket)
            file.write(data)
        self.trainer.model_fft = load_model(f"{self.model_path}/model_fft_re.h5")

    def run(self, rounds, epoch_per_round=1):
        for i in range(rounds):
            print(f"Round {i}")
            self.train_round(epochs=epoch_per_round)
            self.send_weights()
            self.receive_weights()
        self.trainer.evaluation()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # disable GPU usage

    trainer = TrainingWorker(server_ip_port=c.SERVER_IP_PORT,
                             data_path=f"data/{client_c['DATASET_PATH']}_10.csv",  # todo:fix
                             data_cols=client_c['DATASET_COLUMNS'],
                             model_path=client_c['MODEL_PATH'])
    trainer.run(rounds=c.EPOCHS)

