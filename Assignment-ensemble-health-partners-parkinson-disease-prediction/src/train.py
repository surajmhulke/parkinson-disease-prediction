"""
train.py
--------
Train multiple ML models on the Parkinson's dataset and save them.

Usage:
    python src/train.py --data_path dataset/parkinson_disease.csv --seed 42
    python src/train.py --data_path dataset/parkinson_disease.csv --model logistic_regression --seed 42
    python src/train.py --data_path dataset/parkinson_disease.csv --model random_forest --seed 42
    python src/train.py --data_path dataset/parkinson_disease.csv --model xgboost --seed 42
"""

import os
import sys
import pickle
import logging
import argparse
import warnings

from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from data_preprocessing import preprocess, split_and_resample, explore_data, load_data

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
            os.path.join(os.path.dirname(__file__), "..", "outputs", "train.log"),
            mode="w",
            delay=True,
        ),
    ],
)
logger = logging.getLogger("train")

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def get_models(seed: int = 42) -> dict:
    """Return a dict of model_name → unfitted estimator."""
    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=seed
        ),
        "xgboost": XGBClassifier(
            n_estimators=100, eval_metric="logloss", random_state=seed
        ),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model, X_train, y_train, model_name: str = ""):
    """Fit a single model and return it."""
    logger.info("|| %s || Starting training ...", model_name)
    model.fit(X_train, y_train)
    logger.info("|| %s || Training complete.", model_name)
    return model


def save_model(model, name: str, models_dir: str = MODELS_DIR) -> str:
    """Pickle the model; return the saved path."""
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("|| %s || Model saved → %s", name, path)
    return path


def load_model(path: str):
    """Load a pickled model."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train Parkinson's disease classifiers")
    parser.add_argument(
        "--data_path",
        default=os.path.join(os.path.dirname(__file__), "..", "dataset", "parkinson_disease.csv"),
        help="Path to the raw CSV dataset",
    )
    parser.add_argument("--k_features", type=int, default=30, help="Number of top features to select")
    parser.add_argument("--test_size", type=float, default=0.2, help="Validation split fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--model",
        default="all",
        choices=["all", "logistic_regression", "random_forest", "xgboost"],
        help="Model to train",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    logger.info("========================================")
    logger.info("  Parkinson's Disease — Training Run")
    logger.info("========================================")
    logger.info("Args: %s", vars(args))

    # --- Preprocessing ---
    df, feature_names = preprocess(args.data_path, k_features=args.k_features)

    # --- EDA plots ---
    raw_df = load_data(args.data_path)
    explore_data(raw_df)

    # --- Split ---
    X_train, X_val, y_train, y_val = split_and_resample(df, test_size=args.test_size)

    # --- Train ---
    all_models = get_models(seed=args.seed)
    selected = all_models if args.model == "all" else {args.model: all_models[args.model]}

    trained = {}
    model_list = list(selected.items())

    for name, model in tqdm(model_list, desc="Training models", unit="model", ncols=80):
        model = train_model(model, X_train, y_train, model_name=name)
        save_model(model, name)
        trained[name] = model

    logger.info("========================================")
    logger.info("All models trained and saved to outputs/models/")
    logger.info("Run: python src/evaluate.py --data_path %s", args.data_path)
    logger.info("========================================")

    return trained, X_val, y_val, feature_names


if __name__ == "__main__":
    main()
