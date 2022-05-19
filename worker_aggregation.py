import socket
from keras.models import load_model
import numpy as np

from util.socket_functions import send_msg, recv_msg
from util.config import c


class AggregationWorker:
    def __init__(self, ip_port, model_path, num_clients):
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.bind(ip_port)
        self.listener.listen()
        self.epoch = 0
        self.clients_total = num_clients
        self.clients = []
        self.aggregated_data = None
        print(f"AGGREGATIONWORKER: Listening on {ip_port}...")

    def accept_client(self):
        connection, address = self.listener.accept()
        client_id = len(self.clients)
        print(f"\nAGGREGATIONWORKER: New connection {address}")
        self.clients.append(Client(connection, address, client_id))

    def receive(self, client_id):
        client = self.clients[client_id]
        client.data = recv_msg(sock=client.connection)
        print(f"AGGREGATIONWORKER: Received from {client.address}")

    def send_all(self):
        for client in self.clients:
            send_msg(sock=client.connection, msg=self.aggregated_data)

    def aggregate(self):
        # todo: effizienter implementieren (nur weights, keine model)
        # todo: für andere architekturen implementieren (layer-anzahl hier noch fix 7)
        print(f"AGGREGATIONWORKER: Aggregating Epoch {self.epoch}\n")
        weights = []
        weights_new = []
        for client in self.clients:
            with open(f"model/aggregator/client_{client.id}.h5", "wb") as file:
                file.write(client.data)
            model = load_model(f"model/aggregator/client_{client.id}.h5")
            weights.append(model.get_weights())
        for i in range(14):
            layer_weights_all = np.array([weights[x][i] for x in range(len(weights))])
            layer_weights = np.mean(layer_weights_all, axis=0)
            weights_new.append(layer_weights)
        model.set_weights(weights_new)
        model.save(f"model/aggregator/aggregated.h5")
        with open(f"model/aggregator/aggregated.h5", 'rb') as file:
            self.aggregated_data = file.read()
        self.epoch += 1

    def run(self):
        # iterate over the expected number of clients and accept them
        for _ in range(self.clients_total):
            self.accept_client()

        # receive the data from all clients, aggregate the data and send it back to all
        while True:
            for i in range(self.clients_total):  # todo: threads?
                self.receive(client_id=i)
            self.aggregate()
            self.send_all()


class Client:
    def __init__(self, connection, address, client_id):
        self.connection = connection
        self.address = address
        self.id = client_id
        self.data = None
        print(f"New Client - ID:{client_id}, Address:{address} ")


if __name__ == '__main__':
    print("Start of Aggregator")
    aggregator = AggregationWorker(ip_port=c.BIND_IP_PORT, model_path='model/aggregator', num_clients=c.CLIENT_NUM)
    aggregator.run()
