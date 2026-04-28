import joblib, os, yaml, json
import pandas as pd
import numpy as np

from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from typing import Tuple, Dict

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute regression metrics.
    """
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred))
    }

def load_data(
    test_path: str,
    target_column: str
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load test dataset.
    """
    test_df = pd.read_csv(test_path)

    x_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    return x_test, y_test