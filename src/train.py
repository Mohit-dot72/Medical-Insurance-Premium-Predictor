# =============================================================================
# src/train.py
# Model training, evaluation, comparison, and persistence.
# =============================================================================

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model    import LinearRegression
from sklearn.tree            import DecisionTreeRegressor
from sklearn.ensemble        import RandomForestRegressor
from sklearn.metrics         import r2_score, mean_squared_error, mean_absolute_error

# make sure sibling module is importable when run directly
sys.path.insert(0, os.path.dirname(__file__))
from preprocess import build_dataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR    = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR  = os.path.join(ROOT_DIR, "models")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
MODEL_PATH  = os.path.join(MODELS_DIR, "best_model.pkl")

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Candidate models
# ---------------------------------------------------------------------------
def get_candidate_models() -> dict:
    """Return a dict of model_name -> untrained sklearn estimator."""
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree":     DecisionTreeRegressor(random_state=42, max_depth=10),
        "Random Forest":     RandomForestRegressor(
                                n_estimators=200,
                                max_depth=None,
                                min_samples_leaf=2,
                                random_state=42,
                                n_jobs=-1,
                             ),
    }


# ---------------------------------------------------------------------------
# 2. Train & Evaluate
# ---------------------------------------------------------------------------
def train_and_evaluate(X_train, X_test, y_train, y_test, models: dict) -> pd.DataFrame:
    """Fit every model, compute metrics on the held-out test set.

    Metrics computed:
        R2   (coefficient of determination)  -- higher is better
        RMSE (root mean squared error)       -- lower  is better
        MAE  (mean absolute error)           -- lower  is better

    Returns:
        DataFrame with one row per model, columns: Model, R2, RMSE, MAE
    """
    results = []
    trained = {}

    for name, model in models.items():
        print(f"\n[train] Fitting: {name} ...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2   = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)

        results.append({
            "Model": name,
            "R2":    round(r2,   4),
            "RMSE":  round(rmse, 2),
            "MAE":   round(mae,  2),
        })
        trained[name] = model

        print(f"   R2={r2:.4f}  RMSE=${rmse:,.0f}  MAE=${mae:,.0f}")

    results_df = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)
    return results_df, trained


# ---------------------------------------------------------------------------
# 3. Plot comparison
# ---------------------------------------------------------------------------
def plot_model_comparison(results_df: pd.DataFrame, save_path: str = None):
    """Bar chart comparing model R2, RMSE, and MAE scores."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Model Performance Comparison", fontsize=16, fontweight="bold")

    palette = ["#4C9BE8", "#F5A623", "#6DBE6D"]
    metrics = ["R2", "RMSE", "MAE"]
    ylabels = ["R2 Score (higher=better)", "RMSE in $ (lower=better)", "MAE in $ (lower=better)"]

    for ax, metric, ylabel, color in zip(axes, metrics, ylabels, palette):
        bars = ax.bar(results_df["Model"], results_df[metric], color=color, edgecolor="black", linewidth=0.5)
        ax.set_title(metric, fontsize=13)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 0.5,
                    f"{bar.get_height():,.2f}",
                    ha="center", va="center", fontsize=9, color="white", fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[train] Saved comparison plot -> {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# 4. Predicted vs Actual scatter
# ---------------------------------------------------------------------------
def plot_predictions(model, X_test, y_test, model_name: str, save_path: str = None):
    """Scatter plot of predicted vs actual charges for the best model."""
    y_pred = model.predict(X_test)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.4, color="#4C9BE8", edgecolors="none", s=25)
    lims = [min(y_test.min(), y_pred.min()) * 0.95,
            max(y_test.max(), y_pred.max()) * 1.05]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Charges ($)", fontsize=12)
    ax.set_ylabel("Predicted Charges ($)", fontsize=12)
    ax.set_title(f"{model_name} - Actual vs Predicted", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[train] Saved prediction scatter -> {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# 5. Feature importance (Random Forest)
# ---------------------------------------------------------------------------
def plot_feature_importance(model, feature_names, save_path: str = None):
    """Horizontal bar chart of feature importances for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        return
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(importances)))
    importances.plot(kind="barh", ax=ax, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_title("Feature Importances - Random Forest", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[train] Saved feature importance plot -> {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# 6. Save best model
# ---------------------------------------------------------------------------
def save_model(model, feature_names: list, path: str = MODEL_PATH):
    """Serialize model + metadata with Joblib.

    Saves a dict payload:
        {
            "model":         <fitted estimator>,
            "feature_names": [...],
            "model_name":    "...",
        }
    """
    payload = {
        "model":         model,
        "feature_names": feature_names,
        "model_name":    type(model).__name__,
    }
    joblib.dump(payload, path)
    print(f"\n[train] Best model saved -> {path}")


# ---------------------------------------------------------------------------
# 7. Main pipeline
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Medical Insurance Premium - Training Pipeline")
    print("=" * 60)

    # 7a. Load data
    X, y, feature_names, _ = build_dataset()

    # 7b. Train / test split (80 / 20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n[train] Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    # 7c. Train & evaluate all models
    models  = get_candidate_models()
    results_df, trained_models = train_and_evaluate(X_train, X_test, y_train, y_test, models)

    # 7d. Print comparison table
    print("\n" + "=" * 55)
    print("  Model Comparison (sorted by R2, best first)")
    print("=" * 55)
    print(results_df.to_string(index=False))
    print("=" * 55)

    # 7e. Select best model (highest R2)
    best_name  = results_df.iloc[0]["Model"]
    best_model = trained_models[best_name]
    print(f"\n[train] Best model: {best_name} (R2={results_df.iloc[0]['R2']})")

    # 7f. Plots
    plot_model_comparison(results_df,
                          save_path=os.path.join(OUTPUTS_DIR, "model_comparison.png"))
    plot_predictions(best_model, X_test, y_test, best_name,
                     save_path=os.path.join(OUTPUTS_DIR, "predictions_scatter.png"))
    plot_feature_importance(best_model, feature_names,
                            save_path=os.path.join(OUTPUTS_DIR, "feature_importance.png"))

    # 7g. Persist best model
    save_model(best_model, feature_names)

    # 7h. Save metrics CSV for the app to display
    metrics_path = os.path.join(OUTPUTS_DIR, "model_metrics.csv")
    results_df.to_csv(metrics_path, index=False)
    print(f"[train] Metrics saved -> {metrics_path}")

    print("\n[train] Pipeline complete. Outputs saved to /outputs and /models.\n")
    return results_df, best_model, feature_names


if __name__ == "__main__":
    main()
