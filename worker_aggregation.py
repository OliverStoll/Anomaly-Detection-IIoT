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
        self.epoch = 0
        self.current_weights = 0
        self.num_clients = num_clients
        print(f"Listening on {ip_port}...")

    def receive(self):  # currently echos!
        connection, address = self.listener.accept()
        with connection:
            print(f"\nNEW CONNECTION")
            while True:
                data = recv_msg(sock=connection)
                print(f"Receive from {address}")
                self.current_weights += 1
                # while self.current_weights != self.num_clients:
                #    sleep(1)
                send_msg(sock=connection, msg=data)
                self.current_weights = 0

    def add_client(self):
        pass

    def aggregate(self):
        pass

    def run(self):
        c1 = Thread(target=self.receive)
        c2 = Thread(target=self.receive)
        c1.start()
        c2.start()


if __name__ == '__main__':
    aggregator = Aggregator(ip_port=IP_PORT, model_path='results/aggregator', num_clients=2)
    aggregator.run()



