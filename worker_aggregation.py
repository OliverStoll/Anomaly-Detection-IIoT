import socket
from keras.models import load_model
import numpy as np
import pickle

from training import evaluate_model
from models import fft_autoencoder_model

from util.socket_functionality import send_msg, recv_msg
from util.config import c


class AggregationClientObject:
    def __init__(self, connection: object, address: object, client_id: int):
        self.connection = connection
        self.address = address
        self.id = client_id
        self.data = None
        print(f"New Client - ID:{client_id}, Address:{address} ")


class AggregationWorker:
    def __init__(self, ip_port_tuple: (str, int), clients_amount: int,
                 max_iterations: int, model_path: str = "model/aggregator"):
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.bind((ip_port_tuple[0], ip_port_tuple[1]))
        self.listener.listen()
        self.epoch = 0
        self.max_iterations = max_iterations
        self.clients_desired = clients_amount
        self.clients = []
        self.model_path = model_path
        self.aggregated_model_path = f"{self.model_path}/aggregated.h5"
        self.aggregated_data = None
        print(f"AGGREGATIONWORKER: Listening on {ip_port_tuple}...")

    def accept_client(self):
        connection, address = self.listener.accept()
        client_id = len(self.clients)
        print(f"\nAGGREGATIONWORKER: New connection {address}")
        self.clients.append(AggregationClientObject(connection, address, client_id))

    def receive(self, client_id: int):
        client = self.clients[client_id]
        client.data = recv_msg(sock=client.connection)
        print(f"AGGREGATION_WORKER: Received from {client.address}")

    def send_all(self):
        for client in self.clients:
            send_msg(sock=client.connection, msg=self.aggregated_data)

    def aggregate(self):
        # todo: effizienter implementieren (nur weights, keine model)
        # todo: für andere architekturen implementieren (layer-anzahl hier noch fix 7)
        print(f"AGGREGATION_WORKER: Aggregating Epoch {self.epoch}\n")
        client_weights = []
        all_clients_weights = []
        aggregated_weights = []

        # iterate over all clients and save weights from received data
        for client in self.clients:
            client_weights = pickle.loads(client.data)
            all_clients_weights.append(client_weights)

        # average the weights from all clients
        for i in range(len(client_weights)):
            layer_weights_all = np.array([all_clients_weights[x][i] for x in range(len(all_clients_weights))])
            layer_weights_mean = np.mean(layer_weights_all, axis=0)
            aggregated_weights.append(layer_weights_mean)

        # save a model from the averaged weights
        self.aggregated_data = pickle.dumps(aggregated_weights)
        self.epoch += 1

    def run(self):
        # iterate over the expected number of clients and accept them
        for _ in range(self.clients_desired):
            self.accept_client()

        # receive the data from all clients, aggregate the data and send it back to all
        for epoch in range(self.max_iterations):
            for i in range(self.clients_desired):  # todo: threads?
                self.receive(client_id=i)
            self.aggregate()
            self.send_all()


if __name__ == '__main__':
    print("Start of Aggregator")
    aggregator = AggregationWorker(ip_port_tuple=c.LISTEN_IP_PORT,
                                   model_path=c.CLIENT_1['MODEL_PATH'].replace('model/', '/model/federated'),
                                   clients_amount=c.CLIENT_NUM,
                                   max_iterations=100)
    aggregator.run()
