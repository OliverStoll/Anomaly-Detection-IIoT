import yaml
from types import SimpleNamespace
import os

# disable tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import the config from yaml file and convert it to a dot dictionary
config_file = os.environ.get('CONFIG_FILE')
client_name = os.environ.get('CLIENT_NAME')  # client id allows for multiple clients to run the same config
if config_file is None:
    config_file = 'config.yaml'
if client_name is None:
    client_name = 'CLIENT_1'

# import the config file
c = yaml.safe_load(open(f"files/{config_file}"))
c_client = c[client_name]
c = SimpleNamespace(**c)

# print("[CONFIG]", c)
