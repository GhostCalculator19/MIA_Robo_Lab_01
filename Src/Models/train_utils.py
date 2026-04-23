import numpy as np
import pandas as pd
import yaml

def load_data(config_path=None, train_path=None, target_column=None):
    
    # вариант 1: через config (.npy)
    if config_path:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        X = np.load(config['data']['dataset_x_path_np'])
        y = np.load(config['data']['dataset_y_path_np'])
        return X, y

    # вариант 2: через CSV
    elif train_path and target_column:
        df = pd.read_csv(train_path)
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        return X.values, y.values

    else:
        raise ValueError("Передай либо config_path, либо train_path + target_column")