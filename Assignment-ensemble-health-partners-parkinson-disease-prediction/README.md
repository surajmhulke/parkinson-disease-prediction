# Parkinson's Disease Prediction - ML Pipeline

End-to-end machine learning pipeline for predicting Parkinson's disease from voice measurements, with a FastAPI prediction service.

## Dataset

**Direct download:**
```bash
wget -P dataset/ "https://media.geeksforgeeks.org/wp-content/uploads/20250122143413596644/parkinson_disease.csv"
```

Or download manually from the link above and place at `dataset/parkinson_disease.csv`.

The dataset contains 756 voice recordings from 252 patients (188 Parkinson's, 64 healthy), with 754 acoustic features per recording.

## Project Structure

```
ensemble_project/
├── dataset/
│   └── parkinson_disease.csv
├── src/
│   ├── data_preprocessing.py   # EDA, feature selection, splitting
│   ├── train.py                # Model training & saving
│   └── evaluate.py             # Metrics, SHAP plots, recommendation
├── app/
│   └── main.py                 # FastAPI prediction API
├── outputs/
│   ├── models/                 # Saved .pkl model files
│   └── plots/                  # All generated visualisations
├── notebooks/
│   └── Parkinson_Disease_Prediction_using_Machine_Learning.ipynb
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the Pipeline

### 1. Train all models
```bash
cd ensemble_project
source venv/bin/activate

python src/train.py --data_path dataset/parkinson_disease.csv --seed 42
```

Train a specific model:
```bash
python src/train.py --data_path dataset/parkinson_disease.csv --model logistic_regression --seed 42
python src/train.py --data_path dataset/parkinson_disease.csv --model random_forest --seed 42
python src/train.py --data_path dataset/parkinson_disease.csv --model xgboost --seed 42
```

> Available `--model` values: `logistic_regression`, `random_forest`, `xgboost`, `all` (default)

### 2. Evaluate all models
```bash
python src/evaluate.py --data_path dataset/parkinson_disease.csv
```

Evaluate a single saved model:
```bash
python src/evaluate.py --model_path outputs/models/logistic_regression.pkl --data_path dataset/parkinson_disease.csv
```

### 3. Start the prediction API
```bash
uvicorn app.main:app --reload --port 8000
```

Then open: http://localhost:8000/docs (interactive Swagger UI)

### 4. Make a prediction

**Swagger UI (recommended):** open http://localhost:8000/docs → click `/predict` → **Try it out** → paste the JSON below:

```json
{
  "model": "logistic_regression",
  "features": {
    "tqwt_kurtosisValue_dec_34": 2.6202,
    "tqwt_kurtosisValue_dec_28": 1.562,
    "tqwt_kurtosisValue_dec_26": 1.6058,
    "tqwt_kurtosisValue_dec_25": 2.0062,
    "tqwt_maxValue_dec_1": 0.01432,
    "tqwt_minValue_dec_12": -0.024286,
    "tqwt_TKEO_mean_dec_32": 0.000013,
    "tqwt_TKEO_mean_dec_13": 0.000128,
    "tqwt_entropy_log_dec_26": -4459.3054,
    "tqwt_entropy_shannon_dec_35": 0.015562,
    "tqwt_entropy_shannon_dec_17": 360.7126,
    "tqwt_entropy_shannon_dec_11": 4.884,
    "tqwt_entropy_shannon_dec_9": 4.6877,
    "tqwt_energy_dec_33": 0.000002,
    "tqwt_energy_dec_31": 0.000007,
    "tqwt_energy_dec_28": 0.032743,
    "tqwt_energy_dec_27": 0.10807,
    "tqwt_energy_dec_26": 0.057575,
    "tqwt_energy_dec_25": 0.01007,
    "tqwt_energy_dec_16": 0.037555,
    "tqwt_energy_dec_14": 0.012066,
    "tqwt_energy_dec_12": 0.000239,
    "tqwt_energy_dec_7": 0.000164,
    "std_MFCC_8th_coef": 0.17101,
    "mean_MFCC_2nd_coef": 2.4874,
    "IMF_SNR_SEO": 51.6843,
    "VFER_mean": 0.000463,
    "f1": 539.342735,
    "DFA": 0.71826,
    "gender": 1.0
  }
}
```

**cURL equivalent:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"model":"logistic_regression","features":{"tqwt_kurtosisValue_dec_34":2.6202,"tqwt_kurtosisValue_dec_28":1.562,"tqwt_kurtosisValue_dec_26":1.6058,"tqwt_kurtosisValue_dec_25":2.0062,"tqwt_maxValue_dec_1":0.01432,"tqwt_minValue_dec_12":-0.024286,"tqwt_TKEO_mean_dec_32":0.000013,"tqwt_TKEO_mean_dec_13":0.000128,"tqwt_entropy_log_dec_26":-4459.3054,"tqwt_entropy_shannon_dec_35":0.015562,"tqwt_entropy_shannon_dec_17":360.7126,"tqwt_entropy_shannon_dec_11":4.884,"tqwt_entropy_shannon_dec_9":4.6877,"tqwt_energy_dec_33":0.000002,"tqwt_energy_dec_31":0.000007,"tqwt_energy_dec_28":0.032743,"tqwt_energy_dec_27":0.10807,"tqwt_energy_dec_26":0.057575,"tqwt_energy_dec_25":0.01007,"tqwt_energy_dec_16":0.037555,"tqwt_energy_dec_14":0.012066,"tqwt_energy_dec_12":0.000239,"tqwt_energy_dec_7":0.000164,"std_MFCC_8th_coef":0.17101,"mean_MFCC_2nd_coef":2.4874,"IMF_SNR_SEO":51.6843,"VFER_mean":0.000463,"f1":539.342735,"DFA":0.71826,"gender":1.0}}'
```

Get required feature names:
```bash
curl http://localhost:8000/features
```

## Models Trained

| Model | Description |
|---|---|
| `logistic_regression` | Linear classifier with class balancing |
| `random_forest` | 100-tree ensemble with class balancing |
| `xgboost` | Gradient boosted trees |

## Outputs

After running the pipeline, `outputs/` contains:

**Models:**
- `outputs/models/logistic_regression.pkl`
- `outputs/models/random_forest.pkl`
- `outputs/models/xgboost.pkl`

**Plots:**
- `class_distribution.png` - target class balance
- `correlation_heatmap.png` - feature correlations
- `confusion_matrix_<model>.png` - per-model confusion matrix
- `feature_importance_<model>.png` - top-20 feature importances
- `shap_summary_<model>.png` - SHAP beeswarm plots
- `model_comparison.png` - side-by-side metric comparison

## High-Level Approach

1. **Preprocessing:** Patient-level aggregation (mean per patient ID), correlation-based feature pruning (r > 0.7), chi-squared feature selection (top 30 features), MinMaxScaling, RandomOverSampler to balance training classes.

2. **Models:** Logistic Regression (interpretable baseline), Random Forest (non-linear ensemble), XGBoost (gradient boosting).

3. **Evaluation:** Accuracy, F1, ROC-AUC on held-out 20% split. Confusion matrices, feature importance charts, and SHAP summary plots.

4. **Recommendation:** **Logistic Regression** - best generalisation (val ROC-AUC ≈ 0.83), no overfitting, inherently interpretable, produces calibrated probabilities suited for clinical use.
