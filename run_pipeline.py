"""
Standalone script — runs the full ML pipeline end-to-end.
Usage:  python run_pipeline.py [--engineered]
"""
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless — saves PNGs instead of showing windows
import matplotlib.pyplot as plt
import seaborn as sns
import os, joblib

from src.data_loader   import load_data, split_features_target
from src.preprocessing import engineer_features, get_splits, build_scaler_pipeline
from src.models        import get_baseline_models, get_tuning_grid
from src.evaluate      import regression_metrics, cv_score, compare_models

from sklearn.model_selection import RandomizedSearchCV
from sklearn.inspection      import permutation_importance

sns.set_theme(style="whitegrid", palette="muted")
os.makedirs("outputs", exist_ok=True)
os.makedirs("models",  exist_ok=True)
RANDOM_STATE = 42


def main(use_engineered: bool = True) -> None:
    print("─" * 60)
    print("  Boston Housing — ML Pipeline")
    print("─" * 60)

    # ── 1. Load ──────────────────────────────────────────────────
    df = load_data("housing.csv")
    print(f"\n[1] Data loaded: {df.shape[0]} rows × {df.shape[1]} cols")
    print(df.describe().T[["mean", "std", "min", "max"]].round(2))

    # ── 2. Feature Engineering ───────────────────────────────────
    df_proc = engineer_features(df) if use_engineered else df
    X, y    = split_features_target(df_proc)
    X_train, X_test, y_train, y_test = get_splits(X, y)
    print(f"\n[2] Features: {X.shape[1]}  |  Train: {len(y_train)}  Test: {len(y_test)}")

    # ── 3. EDA plots ─────────────────────────────────────────────
    _save_correlation_heatmap(df)
    _save_target_distribution(df)
    print("\n[3] EDA plots saved to outputs/")

    # ── 4. Baseline models ───────────────────────────────────────
    print("\n[4] Baseline cross-validation (5-fold)")
    print(f"{'Model':<22}  {'CV RMSE':>9}  {'CV R2':>7}  {'Test R2':>8}")
    print("─" * 55)

    results = {}
    for name, model in get_baseline_models().items():
        pipeline = build_scaler_pipeline(model)
        cv_res   = cv_score(pipeline, X_train, y_train)
        pipeline.fit(X_train, y_train)
        test_met = regression_metrics(y_test, pipeline.predict(X_test))
        results[name] = {**cv_res,
                         "Test RMSE": test_met["RMSE"],
                         "Test R2":   test_met["R2"],
                         "Test MAE":  test_met["MAE"]}
        print(f"{name:<22}  {cv_res['CV RMSE']:>9.3f}  "
              f"{cv_res['CV R2']:>7.4f}  {test_met['R2']:>8.4f}")

    leaderboard = compare_models(results)
    leaderboard.to_csv("outputs/leaderboard.csv")
    print("\nLeaderboard saved → outputs/leaderboard.csv")

    # ── 5. Hyperparameter Tuning ─────────────────────────────────
    top3       = [m for m in leaderboard.index.tolist()][:3]
    tune_grid  = get_tuning_grid()
    tuned      = {}

    print(f"\n[5] Tuning top-3 models: {top3}")
    for name in top3:
        if name not in tune_grid:
            pipeline = build_scaler_pipeline(get_baseline_models()[name])
            pipeline.fit(X_train, y_train)
            tuned[name] = pipeline
            print(f"  {name}: no grid — using default fit")
            continue
        search = RandomizedSearchCV(
            build_scaler_pipeline(get_baseline_models()[name]),
            tune_grid[name],
            n_iter=20, cv=5,
            scoring="neg_root_mean_squared_error",
            random_state=RANDOM_STATE, n_jobs=-1, verbose=0,
        )
        search.fit(X_train, y_train)
        tuned[name] = search.best_estimator_
        print(f"  {name:<22}  Best CV RMSE={-search.best_score_:.3f}  "
              f"params={search.best_params_}")

    # ── 6. Final evaluation ──────────────────────────────────────
    print("\n[6] Final test-set evaluation")
    print(f"{'Model':<22}  {'RMSE':>7}  {'MAE':>7}  {'R2':>7}  {'MAPE%':>7}")
    print("─" * 55)

    final = {}
    for name, model in tuned.items():
        m = regression_metrics(y_test, model.predict(X_test))
        final[name] = m
        print(f"{name:<22}  {m['RMSE']:>7.3f}  {m['MAE']:>7.3f}  "
              f"{m['R2']:>7.4f}  {m['MAPE%']:>7.2f}%")

    best_name  = min(final, key=lambda k: final[k]["RMSE"])
    best_model = tuned[best_name]
    print(f"\n★  Best model: {best_name}  "
          f"(RMSE={final[best_name]['RMSE']:.3f}, R²={final[best_name]['R2']:.4f})")

    # ── 7. Diagnostics plots ─────────────────────────────────────
    y_pred = best_model.predict(X_test)
    _save_prediction_plot(y_test, y_pred, best_name)
    _save_residual_plot(y_test, y_pred, best_name)
    _save_feature_importance(best_model, X, best_name)
    print("\n[7] Diagnostic plots saved to outputs/")

    # ── 8. Persist ───────────────────────────────────────────────
    path = f"models/best_{best_name.replace(' ', '_').lower()}.pkl"
    joblib.dump(best_model, path)
    print(f"\n[8] Model saved → {path}")
    print("\nDone.")


# ── Helper plot functions ──────────────────────────────────────────────────────

def _save_correlation_heatmap(df: pd.DataFrame) -> None:
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, vmin=-1, vmax=1, ax=ax, annot_kws={"size": 8})
    ax.set_title("Feature Correlation Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig("outputs/correlation_heatmap.png", dpi=120)
    plt.close()


def _save_target_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df["MEDV"], bins=30, color="steelblue", edgecolor="white")
    axes[0].set_title("MEDV Distribution")
    axes[1].hist(np.log(df["MEDV"]), bins=30, color="darkorange", edgecolor="white")
    axes[1].set_title("log(MEDV) Distribution")
    plt.suptitle("Target Variable", fontsize=13)
    plt.tight_layout()
    plt.savefig("outputs/target_distribution.png", dpi=120)
    plt.close()


def _save_prediction_plot(y_true, y_pred, name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(y_true, y_pred, alpha=0.5, s=25, color="steelblue")
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    axes[0].plot([lo, hi], [lo, hi], "r--", linewidth=2)
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predicted")
    axes[0].set_title("Predicted vs Actual")
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5, s=25, color="darkorange")
    axes[1].axhline(0, color="red", linestyle="--", linewidth=2)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Residuals")
    axes[1].set_title("Residuals vs Predicted")
    plt.suptitle(name, fontsize=13)
    plt.tight_layout()
    plt.savefig(f"outputs/predictions_{name.replace(' ', '_').lower()}.png", dpi=120)
    plt.close()


def _save_residual_plot(y_true, y_pred, name: str) -> None:
    from scipy import stats as scipy_stats
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].hist(residuals, bins=25, color="steelblue", edgecolor="white")
    axes[0].set_title("Residual Distribution")
    scipy_stats.probplot(residuals, plot=axes[1])
    axes[1].set_title("Residual Q-Q Plot")
    plt.suptitle(f"Residual Analysis — {name}", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"outputs/residuals_{name.replace(' ', '_').lower()}.png", dpi=120)
    plt.close()


def _save_feature_importance(pipeline, X: pd.DataFrame, name: str) -> None:
    inner = pipeline.named_steps["model"]
    if not hasattr(inner, "feature_importances_"):
        return
    fi_df = pd.DataFrame({
        "Feature":    X.columns,
        "Importance": inner.feature_importances_,
    }).sort_values("Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(fi_df["Feature"][::-1], fi_df["Importance"][::-1],
            color="darkorange", alpha=0.8)
    ax.set_title(f"Feature Importance — {name}", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"outputs/feature_importance_{name.replace(' ', '_').lower()}.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", action="store_true",
                        help="Skip feature engineering, use raw features only")
    args = parser.parse_args()
    main(use_engineered=not args.raw)
