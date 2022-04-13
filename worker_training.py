import socket
import os
from keras.models import load_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from training import *
from scripts.socket_functions import send_msg, recv_msg
from scripts.config import c, client_c


class Trainer:
    """
    This class is responsible for training the model. It encapsulates the training process.
    Additionally, it handles the communication with the aggregation worker and runs the full
    training cycle.

    Attributes:
        model (keras.model): The model to be trained.
    """
    def __init__(self, server_ip_port, data_path, data_cols, model_path):
        self.data, self.data_3d, self.data_train_3d = prepare_data(data_path=data_path, columns=data_cols)
        self.model = init_model(train_data_3d=self.data_train_3d)
        self.model_path = model_path
        self.history = []
        self.epoch = 0
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect(server_ip_port)
        print(f"Connected to {server_ip_port}")

    def train_round(self, epochs=1):
        self.model, history = train_model(model=self.model, data_train_3d=self.data_train_3d, epochs=epochs)
        self.history.append(history)
        self.epoch += epochs

    def send_weights(self):
        self.model.save(f"{self.model_path}/model.h5")
        with open(f"{self.model_path}/model.h5", "rb") as file:
            data = file.read()
        send_msg(sock=self.socket, msg=data)

    def receive_weights(self):
        with open(f"{self.model_path}/model_re.h5", "wb") as file:
            data = recv_msg(sock=self.socket)
            file.write(data)
        self.model = load_model(f"{self.model_path}/model_re.h5")

    def run(self, stop):
        for i in range(stop):
            self.train_round()
            self.send_weights()
            self.receive_weights()
        evaluate_model(model=self.model, data_3d=self.data_3d, history=self.history)


if __name__ == '__main__':
    trainer = Trainer(server_ip_port=c.SERVER_IP_PORT, data_path=client_c['DATASET_PATH'],
                      data_cols=client_c['DATASET_COLUMNS'], model_path=client_c['MODEL_PATH'])
    trainer.run(stop=c.EPOCHS)
