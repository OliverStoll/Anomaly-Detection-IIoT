from scripts.config import c


def scheduler(epoch, lr):
    return lr * c.LR_DECAY