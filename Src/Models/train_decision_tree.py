import joblib, os, yaml, json
import pandas as pd
import numpy as np

from typing import Tuple
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from dvclive import Live

from train_utils import load_data

def main() -> None:

    # Load configuration
    with open('Config/paramts.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Load data
    x_train, y_train = load_data(
        train_path=config["data"]["train_path"],
        target_column=config["data"]["target_column"]
    )

    y_train_log = np.log1p(y_train)

    # Load model params
    params = config['models']['decision_tree']
    
    # Initialize model
    model = DecisionTreeRegressor(**params)

    # Train
    model.fit(x_train, y_train_log)

    # Save tree plot
    os.makedirs(config["reports"]["figures_path"], exist_ok=True)
    plt.figure(figsize=(12, 6))
    
    plot_tree(model, max_depth=2, filled=True)
    plt.savefig(config["reports"]["figures_path"] + "decision_tree_structure.png")
    plt.close()

    # Save model
    os.makedirs(config['models']['models_path'], exist_ok=True)
    joblib.dump(model, config['models']['models_path'] + "decision_tree.pkl")

    # Predict (в лог-пространстве)
    y_pred_log = model.predict(x_train)

    # Возвращаем в исходное пространство
    y_pred = np.expm1(y_pred_log)
    y_true = y_train  # уже в оригинальном масштабе

    # Метрики
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R2: {r2}")
    
    metrics = {
    "train": {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2)
        }
    }

    os.makedirs("dvclive/decision_tree", exist_ok=True)

    with open("dvclive/decision_tree/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()