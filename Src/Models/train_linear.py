import joblib, os, yaml, json
import pandas as pd
import numpy as np

from typing import Tuple
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from dvclive import Live

from train_utils import load_data

def main() -> None:

    # Load configuration
    with open('config/paramts.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Load data
    x_train, y_train = load_data(
        train_path=config["data"]["train_path"],
        target_column=config["data"]["target_column"]
    )

    print(y_train)
    print(y_train.dtype)
    y_train_log = np.log1p(y_train)
    print("LOG OK")
    
    # Load model params
    params = config['models']['linear']
    
    # Initialize model
    model = LinearRegression(**params)

    # Train
    model.fit(x_train, y_train_log)

    # Log coefficients
    coef_dict = {f"coef_{i}": float(v) for i, v in enumerate(model.coef_)}

    with open("Reports/linear_coefficients.json", "w") as f:
        json.dump(coef_dict, f, indent=4)
    
    features = list(coef_dict.keys())
    coefficients = list(coef_dict.values())

    plt.figure(figsize=(8, 5))
    plt.bar(features, coefficients, color='skyblue')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.title("Коэффициенты линейной модели")
    plt.ylabel("Значение коэффициента")
    plt.xlabel("Признаки")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(config["reports"]["figures_path"] + "linear_coefficients.png")
    plt.close()

    # Save model
    os.makedirs(config['models']['models_path'], exist_ok=True)
    joblib.dump(model, config['models']['models_path'] + "linear.pkl")
    
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

    os.makedirs("dvclive/linear", exist_ok=True)

    with open("dvclive/linear/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()