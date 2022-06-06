import os
import socket
from keras.models import load_model
import numpy as np
import pickle
import json
from datetime import datetime
from threading import Thread

from models import fft_autoencoder_model

from util.tcp_messages import send_msg, recv_msg
from util.logs import log_ressource_usage
from util.config import c



class AggregationClientObject:
    def __init__(self, connection: object, address: object, client_id: int):
        self.connection = connection
        self.address = address
        self.id = client_id
        self.data_lstm = None
        self.data_fft = None
        # print(f"New Client - ID:{client_id}, Address:{address} ")


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
        self.aggregated_data_lstm = None
        self.aggregated_data_fft = None
        self.received_bytes = {}
        self.receive_log = f"{c.LOGS_PATH}/data_usage.json"
        # print(f"AGGREGATIONWORKER: Listening on {ip_port_tuple}...")

    def accept_client(self):
        connection, address = self.listener.accept()
        client_id = len(self.clients)
        self.clients.append(AggregationClientObject(connection, address, client_id))

    def receive(self, client_id: int):
        client = self.clients[client_id]
        client.data_lstm = recv_msg(sock=client.connection)
        client.data_fft = recv_msg(sock=client.connection)
        # print(f"AGGREGATION_WORKER: Received from {client.address}")

    def send_all(self):
        for client in self.clients:
            send_msg(sock=client.connection, msg=self.aggregated_data_lstm)
            send_msg(sock=client.connection, msg=self.aggregated_data_fft)

    def log_ressource_usage(self, dict):
        # log the amount of data received as json
        with open(self.receive_log, "w") as f:
            json.dump(dict, f, indent=4)

    def aggregate(self):
        # print(f"AGGREGATION_WORKER: Aggregating Epoch {self.epoch}\n")
        all_clients_weights_lstm = []
        all_clients_weights_fft = []
        aggregated_weights_lstm = []
        aggregated_weights_fft = []

        # iterate over all clients and save weights from received data
        for client in self.clients:
            # print size of received data
            client_weights_lstm = pickle.loads(client.data_lstm)
            client_weights_fft = pickle.loads(client.data_fft)
            all_clients_weights_lstm.append(client_weights_lstm)
            all_clients_weights_fft.append(client_weights_fft)

        if self.epoch == 0:
            self.log_ressource_usage({'lstm': len(client.data_lstm) * self.clients_desired,
                                     'fft_bytes': len(client.data_fft) * self.clients_desired})

        # average the weights from all clients
        for i in range(len(client_weights_lstm)):
            layer_weights_all_lstm = np.array([all_clients_weights_lstm[x][i] for x in range(len(all_clients_weights_lstm))])
            layer_weights_mean_lstm = np.mean(layer_weights_all_lstm, axis=0)
            aggregated_weights_lstm.append(layer_weights_mean_lstm)

        for i in range(len(client_weights_fft)):
            layer_weights_all_fft = np.array([all_clients_weights_fft[x][i] for x in range(len(all_clients_weights_fft))])
            layer_weights_mean_fft = np.mean(layer_weights_all_fft, axis=0)
            aggregated_weights_fft.append(layer_weights_mean_fft)

        # save a model from the averaged weights
        self.aggregated_data_lstm = pickle.dumps(aggregated_weights_lstm)
        self.aggregated_data_fft = pickle.dumps(aggregated_weights_fft)
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

    # start ressource logger thread
    # Thread(target=log_ressource_usage, args=(f"logs/ressources_agg",)).start()

    # print("Start of Aggregator")
    aggregator = AggregationWorker(ip_port_tuple=c.LISTEN_IP_PORT,
                                   model_path=f"model/federated/aggregated/{c.CLIENT_1['MODEL_PATH']}",
                                   clients_amount=c.NUM_CLIENTS,
                                   max_iterations=c.EPOCHS)
    aggregator.run()
