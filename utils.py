
import torch
import pickle
from munch import Munch, munchify
import yaml


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device

def save_pkl(filename, save_object):
	writer = open(filename,'wb')
	pickle.dump(save_object, writer)
	writer.close()

def load_pkl(filename):
	loader = open(filename,'rb')
	file = pickle.load(loader)
	loader.close()
	return file

def load_config(config_path) -> Munch:
    try:
        with open(config_path, 'r') as configuration_fstream:
            yaml_dict = yaml.safe_load(configuration_fstream)
            return munchify(yaml_dict)
    except FileNotFoundError:
        print("Error with config file")
        raise FileNotFoundError
            