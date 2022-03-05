import os


def scheduler(epoch, lr):
    return lr * LR_DECAY


# VARIABLES
SERVER_IP = os.environ['SERVER_IP']
SERVER_PORT = int(os.environ['SERVER_PORT'])
CLIENT_PORT = int(os.environ['CLIENT_PORT'])
EPOCHS = int(os.environ['T_EPOCHS'])
BATCH_SIZE = int(os.environ['T_BATCH_SIZE'])
THRESHOLD = float(os.environ['T_THRESHOLD'])
SPLIT = int(os.environ['T_SPLIT'])
LEARNING_RATE = float(os.environ['T_LEARNING_RATE'])
LR_DECAY = float(os.environ['T_LR_DECAY'])
LAYERS_EXPONENT = int(os.environ['T_LAYERS_EXPONENT'])
DATASET = int(os.environ['T_DATASET'])
LOSS = 'mse'  # 'mean_squared_logarithmic_error'
IP_PORT = (SERVER_IP, SERVER_PORT)
print(f"SERVER_IP_PORT: {IP_PORT}, EPOCHS: {EPOCHS}")
