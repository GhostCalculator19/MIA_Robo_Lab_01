import numpy as np
import yaml

def load_data(config_path):
    #with open(config_path, "r") as f:
    #    config = yaml.safe_load(f)
    X, y = load_data("Config/paramts.yaml")
    #X = np.load(config['data']['dataset_x_path_np'])
    #y = np.load(config['data']['dataset_y_path_np'])

    return X, y