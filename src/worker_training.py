from datetime import datetime
import socket
import pickle
import os
from time import sleep


from training import Training
from util.tcp_messages import send_msg, recv_msg
from util.config import c, config

start = datetime.now()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sleep_times = {"bearing_experiment-1": 35, "bearing_experiment-2": 22, "bearing_experiment-3": 90}


class TrainingWorker:
    """
    This worker class is responsible for training the model.

    It encapsulates the training process.
    Additionally, it handles the communication with the aggregation worker and runs the full
    training cycle.
    """

    def __init__(self, connect_ip_port: (str, int), experiment_name: str, train_cols: list, client_name: str):
        if os.getenv("CLIENT_NAME") == "CLIENT_0" and os.getenv("TRANSFER_LEARNING"):
            self.trainer = Training(experiment_name=experiment_name, load_columns=[0, 1], train_columns=[1],
                                    model_type="federated")
        else:
            self.trainer = Training(experiment_name=experiment_name, load_columns=train_cols, train_columns=[0],
                                    model_type="federated")
        self.trainer.verbose = 1
        self.epoch = 0
        self.client_name = client_name
        self.experiment_name = experiment_name
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(f"TrainingWorker: Connecting to {connect_ip_port}...")
        try:
            self.socket.connect((connect_ip_port[0], connect_ip_port[1]))
        except ConnectionRefusedError:
            print("TrainingWorker: ERROR... trying via docker...")
            try:
                self.socket.connect(("host.docker.internal", connect_ip_port[1]))
            except ConnectionRefusedError:
                print("TrainingWorker: host.docker.interal FAILED")
                exit(1)

    def train_round(self, epochs):
        self.trainer.train_models(epochs=epochs)
        self.epoch += epochs

    def send_weights(self):
        weights_lstm = self.trainer.model_lstm.get_weights()
        weights_fft = self.trainer.model_fft.get_weights()
        weights_lstm_data = pickle.dumps(weights_lstm)
        weights_fft_data = pickle.dumps(weights_fft)
        send_msg(sock=self.socket, msg=weights_lstm_data)
        send_msg(sock=self.socket, msg=weights_fft_data)

    def receive_weights(self):
        weights_data_lstm = recv_msg(sock=self.socket)
        weights_data_fft = recv_msg(sock=self.socket)
        weights_lstm = pickle.loads(weights_data_lstm)
        weights_fft = pickle.loads(weights_data_fft)
        self.trainer.model_lstm.set_weights(weights_lstm)
        self.trainer.model_fft.set_weights(weights_fft)

    def run(self, rounds, epochs_per_round):
        client_id = int(self.client_name.split("_")[1])
        sleep_secs = client_id * sleep_times[self.experiment_name]
        for i in range(rounds):
            print(f"Round {i}")
            if not os.getenv("DOCKER_MODE"):
                print(f"Waiting {sleep_secs}s for other workers to finish their training...")
                sleep(sleep_secs)
            self.train_round(epochs=epochs_per_round)
            self.send_weights()
            self.receive_weights()

        if not os.getenv("DOCKER_MODE"):
            print(f"Waiting {sleep_secs}s for other workers to finish their training...")
            sleep(sleep_secs)
        self.trainer.save_results()


if __name__ == '__main__':

    # EVALUATE TRAINING AS DOCKER IMAGE
    for path, subdirs, files in os.walk("./data"):
        for name in files:
            print(os.path.join(path, name))

    print(f"TRAINING_WORKER: Starting as {os.environ.get('CLIENT_NAME')} ")
    ip = os.getenv("IP", c.CONNECT_IP_PORT[0])
    port = int(os.getenv("PORT", c.CONNECT_IP_PORT[1]))
    epochs = int(os.getenv("EPOCHS", c.EPOCHS))
    connect_ip_port = (ip, port)
    client_config = config[os.getenv("CLIENT_NAME")]
    experiment_name = os.getenv("EXPERIMENT_NAME", config['EXPERIMENT_NAME'])
    train_columns = os.getenv("TRAIN_COLUMN", client_config[experiment_name])

    trainer = TrainingWorker(connect_ip_port=connect_ip_port,
                             experiment_name=experiment_name,
                             train_cols=train_columns,
                             client_name=os.environ.get('CLIENT_NAME'))
    trainer.run(rounds=epochs, epochs_per_round=c.EPOCHS_PER_ROUND)


