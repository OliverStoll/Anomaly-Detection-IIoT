from functionality.config import c


def scheduler(epoch, lr):
    return lr * (1-c.LR_DECAY)

