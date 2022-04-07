import yaml
from types import SimpleNamespace


# import the config from yaml file and convert it to a dot dictionary called c
c = yaml.safe_load(open("configs/config.yaml"))
c = SimpleNamespace(**c)