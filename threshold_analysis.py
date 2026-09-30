import json
import os

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

from src.preprocessing import prepare_features


# =========================================================
# CONFIGURATION
# =========================================================

DATA_PATH = "data/diabetic_data.csv"

SAMPLE_SIZE = 20000
RANDOM_STATE = 42

MODEL_DIR = "models"

OUTPUT_DIR = "evaluation_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# 1. LOAD SAME DATASET
# =========================================================

print("=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

df, _ = train_test_split(
    df,
    train_size=SAMPLE_SIZE,
    stratify=df["readmitted"],
    random_state=RANDOM_STATE
)

df = df.reset_index(drop=True)

X, y = prepare_features(df)


# =========================================================
# 2. TRAIN / VALIDATION / TEST SPLIT
# =========================================================

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

X_train, X_validation, y_train, y_validation = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    stratify=y_train_full,
    random_state=RANDOM_STATE
)

print("\nTraining:", len(X_train))
print("Validation:", len(X_validation))
print("Test:", len(X_test))


# =========================================================
# 3. LOAD MODELS
# =========================================================

models = {
    "Optimized Logistic Regression":
        "optimized_logistic_regression_pipeline.pkl",

    "Optimized Random Forest":
        "optimized_random_forest_pipeline.pkl"
}


# =========================================================
# 4. THRESHOLD RANGE
# =========================================================

thresholds = np.arange(
    0.10,
    0.91,
    0.05
)


all_results = []
best_thresholds = {}


# =========================================================
# 5. ANALYZE EACH MODEL
# =========================================================

for model_name, filename in models.items():

    path = os.path.join(
        MODEL_DIR,
        filename
    )

    if not os.path.exists(path):
        print(
            f"\nSkipping {model_name}: "
            f"{filename} not found."
        )
        continue

    print("\n" + "-" * 60)
    print(model_name)
    print("-" * 60)

    model = joblib.load(path)

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    model_results = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_validation,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_validation,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_validation,
            predictions,
            zero_division=0
        )

        accuracy = accuracy_score(
            y_validation,
            predictions
        )

        result = {
            "model": model_name,
            "threshold": float(threshold),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }

        model_results.append(result)
        all_results.append(result)


    # -----------------------------------------------------
    # Select threshold with highest validation F1
    # -----------------------------------------------------

    best = max(
        model_results,
        key=lambda x: x["f1"]
    )

    best_thresholds[model_name] = best["threshold"]

    print(
        f"Best threshold: "
        f"{best['threshold']:.2f}"
    )

    print(
        f"Validation F1: "
        f"{best['f1']:.4f}"
    )

    print(
        f"Validation precision: "
        f"{best['precision']:.4f}"
    )

    print(
        f"Validation recall: "
        f"{best['recall']:.4f}"
    )


# =========================================================
# 6. SAVE THRESHOLD RESULTS
# =========================================================

results_df = pd.DataFrame(
    all_results
)

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "threshold_analysis.csv"
    ),
    index=False
)


# =========================================================
# 7. SAVE BEST THRESHOLDS
# =========================================================

# Convert names into app-friendly keys.

app_thresholds = {
    "Logistic Regression":
        best_thresholds.get(
            "Optimized Logistic Regression",
            0.50
        ),

    "Random Forest":
        best_thresholds.get(
            "Optimized Random Forest",
            0.50
        ),

    # Existing final models will use 0.50
    # until their validation threshold is
    # explicitly analyzed.
    "RBF SVM": 0.50,
    "XGBoost": 0.50,
    "Stacking Ensemble": 0.50
}


with open(
    os.path.join(
        MODEL_DIR,
        "thresholds.json"
    ),
    "w"
) as file:

    json.dump(
        app_thresholds,
        file,
        indent=4
    )


print("\n" + "=" * 70)
print("SELECTED THRESHOLDS")
print("=" * 70)

for model, threshold in app_thresholds.items():

    print(
        f"{model}: {threshold:.2f}"
    )


print("\nSaved:")
print(
    "evaluation_results/threshold_analysis.csv"
)

print(
    "models/thresholds.json"
)

print("\nThreshold analysis completed.")