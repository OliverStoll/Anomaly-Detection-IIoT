import yaml
from types import SimpleNamespace
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import the config from yaml file and convert it to a dot dictionary
config_file = os.getenv('CONFIG_FILE', 'config.yaml')
client_name = os.environ.get('CLIENT_NAME', 'CLIENT_0')  # client id allows for multiple clients to run the same config
config_data = yaml.safe_load(open(f"configs/{config_file}"))
client_config = config_data[client_name]

new_config = {}
for key, value in config_data.items():
    type_of_value = type(value)
    new_config[key] = os.getenv(key, value)

# config = new_config
config = SimpleNamespace(**config_data)
