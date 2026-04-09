"""
data_preprocessing.py
---------------------
Handles loading, EDA, feature selection, and splitting for Parkinson's dataset.
"""

import os
import json
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_data(data_path: str) -> pd.DataFrame:
    """Load raw CSV and return a DataFrame."""
    df = pd.read_csv(data_path)
    logger.info("Loaded data: %d rows × %d columns from '%s'", df.shape[0], df.shape[1], data_path)
    return df


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------

def explore_data(df: pd.DataFrame) -> None:
    """Print basic EDA statistics and save distribution plots."""
    logger.info("=== EDA ===")
    logger.info("Shape: %s", df.shape)
    logger.info("Total missing values: %d", df.isnull().sum().sum())
    logger.info("Class distribution:\n%s", df["class"].value_counts().to_string())

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Class distribution pie
    fig, ax = plt.subplots()
    counts = df["class"].value_counts()
    ax.pie(counts.values, labels=["Parkinson (1)", "Healthy (0)"], autopct="%1.1f%%")
    ax.set_title("Class Distribution")
    fig.savefig(os.path.join(PLOTS_DIR, "class_distribution.png"), bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: outputs/plots/class_distribution.png")

    # Correlation heatmap on a small subset
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:20]
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(df[numeric_cols].corr(), ax=ax, cmap="coolwarm", center=0)
    ax.set_title("Feature Correlation (first 20 features)")
    fig.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: outputs/plots/correlation_heatmap.png")


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def remove_correlated_features(df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """Drop highly correlated features (|r| > threshold)."""
    original_cols = df.shape[1]
    columns = list(df.columns)
    for col in list(columns):
        if col == "class" or col not in df.columns:
            continue
        filtered_columns = [col]
        for col1 in df.columns:
            if col1 == col or col1 == "class":
                continue
            val = df[col].corr(df[col1])
            if val > threshold:
                if col1 in columns:
                    columns.remove(col1)
            else:
                filtered_columns.append(col1)
        filtered_columns = [c for c in filtered_columns if c in df.columns]
        if "class" not in filtered_columns:
            filtered_columns.append("class")
        df = df[filtered_columns]
    logger.info(
        "Correlation filter (threshold=%.1f): %d → %d columns",
        threshold, original_cols, df.shape[1],
    )
    return df


def select_k_best_features(df: pd.DataFrame, k: int = 30) -> tuple[pd.DataFrame, list]:
    """Select top-k features using chi-squared test."""
    X = df.drop("class", axis=1)
    y = df["class"]

    X_norm = MinMaxScaler().fit_transform(X)
    selector = SelectKBest(chi2, k=k)
    selector.fit(X_norm, y)

    mask = selector.get_support()
    selected_features = X.columns[mask].tolist()

    df_selected = X.loc[:, mask].copy()
    df_selected["class"] = y.values
    logger.info("SelectKBest (k=%d): kept %d features → shape %s", k, k, df_selected.shape)
    return df_selected, selected_features


def preprocess(data_path: str, k_features: int = 30) -> tuple[pd.DataFrame, list]:
    """
    Full preprocessing pipeline:
      1. Load raw data
      2. Aggregate per patient (group by id)
      3. Remove highly correlated features
      4. Select top-k features via chi2
    Returns (preprocessed_df, selected_feature_names).
    """
    logger.info("=== Preprocessing pipeline started ===")
    df = load_data(data_path)

    df = df.groupby("id").mean().reset_index()
    df.drop("id", axis=1, inplace=True)
    logger.info("After patient-level aggregation (groupby id): shape=%s", df.shape)

    df = remove_correlated_features(df)
    df, feature_names = select_k_best_features(df, k=k_features)

    feature_path = os.path.join(OUTPUTS_DIR, "feature_names.json")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(feature_path, "w") as f:
        json.dump(feature_names, f)
    logger.info("Feature names saved → %s", feature_path)

    logger.info("=== Preprocessing pipeline complete ===")
    return df, feature_names


# ---------------------------------------------------------------------------
# Splitting & resampling
# ---------------------------------------------------------------------------

def split_and_resample(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 10,
) -> tuple:
    """
    Train/val split + RandomOverSampler on training set.
    Returns (X_train, X_val, y_train, y_val).
    """
    features = df.drop("class", axis=1)
    target = df["class"]

    X_train, X_val, y_train, y_val = train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )

    ros = RandomOverSampler(sampling_strategy=1.0, random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

    logger.info(
        "Split: train(resampled)=%s  val=%s  | class balance: %s",
        X_resampled.shape,
        X_val.shape,
        dict(pd.Series(y_resampled).value_counts()),
    )
    return X_resampled, X_val, y_resampled, y_val
