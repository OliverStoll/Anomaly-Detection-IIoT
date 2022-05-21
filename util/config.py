import yaml
from types import SimpleNamespace
import os

# disable tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import the config from yaml file and convert it to a dot dictionary
config_file = os.environ.get('CONFIG_FILE')
client = os.environ.get('CLIENT_ID')  # client id allows for multiple clients to run the same config
if config_file is None:
    config_file = 'config.yaml'
    client = 'CLIENT_1'

# import the config file
c = yaml.safe_load(open(f"files/training/{config_file}"))
c_client = c[client]
c = SimpleNamespace(**c)

# print("[CONFIG]", c)
