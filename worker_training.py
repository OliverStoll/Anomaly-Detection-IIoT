import socket
import os
from keras.models import load_model

from training import lstm_autoencoder_model, fft_autoencoder_model, init_models, train_models, load_and_normalize_data,\
    calculate_fft_from_data
from evaluation import evaluate_model_lstm, evaluate_model_fft
from functionality.socket_functions import send_msg, recv_msg
from functionality.config import c, client_c

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Trainer:
    """
    This worker class is responsible for training the model.

    It encapsulates the training process.
    Additionally, it handles the communication with the aggregation worker and runs the full
    training cycle.
    """

    def __init__(self, server_ip_port, data_path, data_cols, model_path):
        self.data_3d, self.data_train_3d = load_and_normalize_data(data_path=data_path, columns=data_cols)
        self.fft_data_train_3d = calculate_fft_from_data(data_3d=self.data_train_3d)
        self.model_lstm, self.model_fft = init_models(train_data_3d=self.data_train_3d)
        self.model_path = model_path
        self.history_lstm = []
        self.history_fft = []
        self.epoch = 0
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.socket.connect(server_ip_port)
        print(f"Connected to {server_ip_port}")

    def train_round(self, epochs=1):
        # TODO: check efficiency of two models being trained simoultaneously
        self.model_lstm, self.model_fft, history_lstm, history_fft = train_models(model_lstm=self.model_lstm,
                                                                                  model_fft=self.model_fft,
                                                                                  data_train_3d=self.data_train_3d,
                                                                                  fft_train_3d=self.fft_data_train_3d,
                                                                                  epochs=epochs)
        self.history_lstm.append(history_lstm)
        self.history_fft.append(history_fft)
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
            # self.send_weights()
            # self.receive_weights()
        evaluate_model_lstm(model=self.model, data_3d=self.data_3d, history=self.history_lstm)


if __name__ == '__main__':
    trainer = Trainer(server_ip_port=c.SERVER_IP_PORT, data_path="data/" + client_c['DATASET_PATH']+'_10.csv',  # todo:fix
                      data_cols=client_c['DATASET_COLUMNS'], model_path=client_c['MODEL_PATH'])
    trainer.run(stop=c.EPOCHS)

