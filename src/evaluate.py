import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate


def regression_metrics(y_true, y_pred) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE%": mape}


def cv_score(pipeline, X, y, cv: int = 5) -> dict:
    scoring = {
        "neg_rmse": "neg_root_mean_squared_error",
        "neg_mae":  "neg_mean_absolute_error",
        "r2":       "r2",
    }
    results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, return_train_score=True)
    return {
        "CV RMSE":       -results["test_neg_rmse"].mean(),
        "CV RMSE std":   results["test_neg_rmse"].std(),
        "CV MAE":        -results["test_neg_mae"].mean(),
        "CV R2":         results["test_r2"].mean(),
        "Train R2":      results["train_r2"].mean(),
    }


def compare_models(results: dict) -> pd.DataFrame:
    rows = []
    for name, metrics in results.items():
        rows.append({"Model": name, **metrics})
    df = pd.DataFrame(rows).set_index("Model")
    return df.sort_values("CV RMSE")
