import os


def scheduler(epoch, lr):
    return lr * LR_DECAY


# VARIABLES
SERVER_IP = os.environ['SERVER_IP']
SERVER_PORT = int(os.environ['SERVER_PORT'])
CLIENT_PORT = int(os.environ['CLIENT_PORT'])
EPOCHS = int(os.environ['EPOCHS'])
BATCH_SIZE = 16
THRESHOLD = 0.1
SPLIT = 10
LEARNING_RATE = 5e-4
LR_DECAY = 0.98
LAYERS_EXPONENT = 4
LOSS = 'mse'  # 'mean_squared_logarithmic_error'
DATASET = 2
IP_PORT = (SERVER_IP, SERVER_PORT)
print(f"IP_PORT: {IP_PORT}, CLIENT_PORT: {CLIENT_PORT}, EPOCHS: {EPOCHS}")
