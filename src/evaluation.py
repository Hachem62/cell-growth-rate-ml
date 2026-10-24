import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate

from src.preprocessing import RANDOM_STATE

SCORING = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}


def cross_validate_candidates(candidates: dict, X_train, y_train, n_splits: int = 5) -> pd.DataFrame:
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, info in candidates.items():
        scores = cross_validate(info["model"], X_train, y_train, cv=cv,
                                scoring=SCORING, return_train_score=True)
        rows.append({
            "Model": name,
            "Family": info["family"],
            "Degree": info["degree"],
            "k": info["k"],
            "Weights": info["weights"],
            "Train RMSE": -scores["train_rmse"].mean(),
            "CV RMSE": -scores["test_rmse"].mean(),
            "CV RMSE Std": scores["test_rmse"].std(),
            "CV MAE": -scores["test_mae"].mean(),
            "CV R2": scores["test_r2"].mean(),
        })
    return pd.DataFrame(rows).sort_values("CV RMSE").reset_index(drop=True)


def evaluate_on_test(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    return {
        "Test RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "Test MAE": float(mean_absolute_error(y_test, y_pred)),
        "Test R2": float(r2_score(y_test, y_pred)),
    }