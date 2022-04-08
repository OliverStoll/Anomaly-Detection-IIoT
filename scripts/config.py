import yaml
from types import SimpleNamespace
import os

# disable tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import the config from yaml file and convert it to a dot dictionary called c
c = yaml.safe_load(open("configs/config.yaml"))
c = SimpleNamespace(**c)