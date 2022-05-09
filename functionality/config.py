import yaml
from types import SimpleNamespace
import os

# disable tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import the config from yaml file and convert it to a dot dictionary called c
config_file = os.environ.get('CONFIG_FILE')
client = os.environ.get('CLIENT_ID')
if config_file is None:
    config_file = 'config.yaml'
    client = 'CLIENT_1'
# import the config file
c = yaml.safe_load(open(f"configs/training/{config_file}"))
client_c = c[client]
c = SimpleNamespace(**c)

print(c)