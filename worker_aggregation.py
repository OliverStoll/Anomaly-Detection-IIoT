import socket
from threading import Thread
from config import *
from socket_functions import *
from time import sleep


class Aggregator:
    def __init__(self, ip_port, model_path, num_clients):
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.bind(ip_port)
        self.listener.listen()
        print(f"Listening on {ip_port}...")
        self.epoch = 0
        self.clients_total = num_clients
        self.clients = []
        self.weights_received = 0

        self.data = None

    def receive(self):  # currently echos!
        connection, address = self.listener.accept()
        print(f"\nNEW CONNECTION")
        with connection:
            while True:
                data = recv_msg(sock=connection)
                print(f"Receive from {address}")
                self.weights_received += 1
                # TODO signal all threads to now send back weights (maybe in one function?)
                while self.weights_received != self.clients_total:
                    print(f"Weights: {self.weights_received}/{self.clients_total} ...")
                    sleep(5)
                send_msg(sock=connection, msg=data)
                sleep(7)
                self.weights_received = 0

    def send(self):
        pass

    def accept_client(self):
        connection, address = self.listener.accept()

    def aggregate(self):
        pass

    def run(self):
        c1 = Thread(target=self.receive)
        c2 = Thread(target=self.receive)
        c1.start()
        c2.start()


if __name__ == '__main__':
    print("Start of Aggregator")
    aggregator = Aggregator(ip_port=IP_PORT, model_path='results/aggregator', num_clients=CLIENT_NUM)
    aggregator.run()
