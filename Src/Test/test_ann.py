import joblib, os, yaml, json, datetime
import pandas as pd
import numpy as np

from typing import Tuple, Dict
import tensorflow as tf
from dvclive import Live

from test_utils import load_data, compute_metrics


def main() -> None:

    # Load configuration
    with open("Config/paramts.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    # Load data
    x_test, y_test = load_data(
        test_path=config["data"]["test_path"],
        target_column=config["data"]["target_column"]
    )
    
    
    # Load model
    model = tf.keras.models.load_model(config['models']['models_path'] + "ann.keras")

    with Live(dir="dvclive/ann", save_dvc_exp=True) as live:

        # Predict
        y_pred = np.expm1(model.predict(x_test).flatten())   # inverse log transform

        # Metrics
        metrics = compute_metrics(y_test.values, y_pred)

        for metric, value in metrics.items():
            live.log_metric(f"test/{metric}", value)


if __name__ == "__main__":
    main()