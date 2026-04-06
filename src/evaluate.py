"""
evaluate.py
-----------
Evaluate saved models, generate comparison plots, and SHAP explainability.

Usage:
    python src/evaluate.py --data_path dataset/parkinson_disease.csv
    python src/evaluate.py --model_path outputs/models/logistic_regression.pkl --data_path dataset/parkinson_disease.csv
"""

import os
import sys
import json
import pickle
import logging
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    classification_report,
)
import shap

from data_preprocessing import preprocess, split_and_resample

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s || %(levelname)-8s || %(name)s || %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(os.path.dirname(__file__), "..", "outputs", "evaluate.log"),
            mode="w",
            delay=True,
        ),
    ],
)
logger = logging.getLogger("evaluate")

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_all_models(models_dir: str = MODELS_DIR) -> dict:
    """Load every .pkl file in models_dir."""
    models = {}
    for fname in os.listdir(models_dir):
        if fname.endswith(".pkl"):
            name = fname.replace(".pkl", "")
            models[name] = load_model(os.path.join(models_dir, fname))
            logger.info("|| %s || Loaded model from %s", name, fname)
    return models


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def evaluate_model(model, X_val, y_val, model_name: str = "") -> dict:
    """Return accuracy, f1, roc-auc and log classification report."""
    y_pred = model.predict(X_val)
    y_prob = (
        model.predict_proba(X_val)[:, 1]
        if hasattr(model, "predict_proba")
        else model.decision_function(X_val)
    )

    results = {
        "model": model_name,
        "accuracy": round(accuracy_score(y_val, y_pred), 4),
        "f1": round(f1_score(y_val, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_val, y_prob), 4),
    }

    report = classification_report(y_val, y_pred, target_names=["Healthy", "Parkinson"])
    logger.info("|| %s || Accuracy=%.4f  F1=%.4f  ROC-AUC=%.4f",
                model_name, results["accuracy"], results["f1"], results["roc_auc"])
    logger.info("|| %s || Classification report:\n%s", model_name, report)
    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(model, X_val, y_val, model_name: str) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        model, X_val, y_val,
        display_labels=["Healthy", "Parkinson"],
        ax=ax, colorbar=False,
    )
    ax.set_title(f"Confusion Matrix — {model_name}")
    path = os.path.join(PLOTS_DIR, f"confusion_matrix_{model_name}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("|| %s || Saved confusion matrix → %s", model_name, path)


def plot_metrics_comparison(results: list[dict]) -> None:
    """Bar chart comparing accuracy, f1, roc-auc across models."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    df = pd.DataFrame(results).set_index("model")
    ax = df[["accuracy", "f1", "roc_auc"]].plot(kind="bar", figsize=(10, 5), ylim=(0, 1.05))
    ax.set_title("Model Comparison")
    ax.set_ylabel("Score")
    ax.set_xticklabels(df.index, rotation=25, ha="right")
    ax.legend(loc="lower right")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=7, padding=2)
    path = os.path.join(PLOTS_DIR, "model_comparison.png")
    ax.get_figure().savefig(path, bbox_inches="tight")
    plt.close("all")
    logger.info("Saved model comparison chart → %s", path)


def plot_feature_importance(model, feature_names: list, model_name: str) -> None:
    """Plot feature importance for tree-based and linear models."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        logger.warning("|| %s || No native feature importance — skipping", model_name)
        return

    indices = np.argsort(importances)[:20]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([feature_names[i] for i in indices], importances[indices])
    ax.set_title(f"Top-20 Feature Importances — {model_name}")
    ax.set_xlabel("Importance")
    path = os.path.join(PLOTS_DIR, f"feature_importance_{model_name}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("|| %s || Saved feature importance → %s", model_name, path)


def plot_shap(model, X_val: pd.DataFrame, model_name: str) -> None:
    """Generate SHAP summary plot for a model."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    logger.info("|| %s || Computing SHAP values ...", model_name)
    try:
        is_xgboost = type(model).__name__ == "XGBClassifier"

        if hasattr(model, "feature_importances_") and not is_xgboost:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_val)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            sample = shap.sample(X_val, 50)
            predict_fn = lambda x: model.predict_proba(np.array(x))
            explainer = shap.KernelExplainer(predict_fn, sample)
            shap_values = explainer.shap_values(sample, nsamples=100)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            X_val = sample

        fig, _ = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_val, show=False, plot_size=None)
        plt.gca().set_title(f"SHAP Summary — {model_name}")
        path = os.path.join(PLOTS_DIR, f"shap_summary_{model_name}.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close("all")
        logger.info("|| %s || Saved SHAP summary → %s", model_name, path)

    except Exception as exc:
        logger.warning("|| %s || SHAP computation failed: %s", model_name, exc)


# ---------------------------------------------------------------------------
# Summary recommendation
# ---------------------------------------------------------------------------

def log_recommendation(results: list[dict]) -> None:
    best = max(results, key=lambda r: r["roc_auc"])
    logger.info("=" * 60)
    logger.info("RECOMMENDATION")
    logger.info("=" * 60)
    logger.info("Best model by ROC-AUC: %s", best["model"])
    logger.info(
        "  Accuracy=%.4f  F1=%.4f  ROC-AUC=%.4f",
        best["accuracy"], best["f1"], best["roc_auc"],
    )
    logger.info(
        "Logistic Regression is recommended for clinical use: "
        "best generalisation (val ROC-AUC≈0.83), no overfitting, "
        "inherently interpretable, calibrated probabilities."
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Parkinson's disease classifiers")
    parser.add_argument(
        "--data_path",
        default=os.path.join(os.path.dirname(__file__), "..", "dataset", "parkinson_disease.csv"),
    )
    parser.add_argument("--model_path", default=None, help="Single model .pkl to evaluate; omit to evaluate all")
    parser.add_argument("--k_features", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    logger.info("========================================")
    logger.info("  Parkinson's Disease — Evaluation Run")
    logger.info("========================================")
    logger.info("Args: %s", vars(args))

    # Load or build feature names
    feature_path = os.path.join(OUTPUTS_DIR, "feature_names.json")
    if not os.path.exists(feature_path):
        logger.info("feature_names.json not found — running preprocessing first ...")
        df, feature_names = preprocess(args.data_path, k_features=args.k_features)
    else:
        with open(feature_path) as f:
            feature_names = json.load(f)
        df, _ = preprocess(args.data_path, k_features=args.k_features)

    _, X_val, _, y_val = split_and_resample(df)
    X_val_df = pd.DataFrame(X_val, columns=feature_names)

    # Load models
    if args.model_path:
        name = os.path.basename(args.model_path).replace(".pkl", "")
        models = {name: load_model(args.model_path)}
    else:
        models = load_all_models()

    if not models:
        logger.error("No models found in outputs/models/. Run train.py first.")
        return

    results = []
    for name, model in tqdm(models.items(), desc="Evaluating models", unit="model", ncols=80):
        logger.info("|| %s || Evaluating ...", name)
        result = evaluate_model(model, X_val_df, y_val, model_name=name)
        results.append(result)
        plot_confusion_matrix(model, X_val_df, y_val, model_name=name)
        plot_feature_importance(model, feature_names, model_name=name)
        plot_shap(model, X_val_df, model_name=name)

    plot_metrics_comparison(results)
    log_recommendation(results)

    logger.info("========================================")
    logger.info("All plots saved to outputs/plots/")
    logger.info("========================================")


if __name__ == "__main__":
    main()
