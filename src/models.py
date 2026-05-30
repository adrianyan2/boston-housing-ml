from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


def get_baseline_models() -> dict:
    models = {
        "Linear Regression":       LinearRegression(),
        "Ridge":                   Ridge(alpha=1.0),
        "Lasso":                   Lasso(alpha=0.1),
        "ElasticNet":              ElasticNet(alpha=0.1, l1_ratio=0.5),
        "KNN":                     KNeighborsRegressor(n_neighbors=5),
        "SVR":                     SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1),
        "Random Forest":           RandomForestRegressor(n_estimators=100, random_state=42),
        "Extra Trees":             ExtraTreesRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting":       GradientBoostingRegressor(n_estimators=200, random_state=42),
    }
    if _HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, random_state=42,
            verbosity=0, eval_metric="rmse",
        )
    return models


def get_tuning_grid() -> dict:
    grid = {
        "Ridge": {
            "model__alpha": [0.01, 0.1, 1, 10, 100, 500]
        },
        "Lasso": {
            "model__alpha": [0.001, 0.01, 0.1, 0.5, 1, 5]
        },
        "Random Forest": {
            "model__n_estimators":  [100, 200],
            "model__max_depth":     [None, 10, 20],
            "model__min_samples_split": [2, 5],
        },
        "Gradient Boosting": {
            "model__n_estimators":  [100, 200],
            "model__learning_rate": [0.05, 0.1, 0.2],
            "model__max_depth":     [3, 4, 5],
        },
    }
    if _HAS_XGB:
        grid["XGBoost"] = {
            "model__n_estimators":  [100, 200],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth":     [3, 4, 6],
        }
    return grid
