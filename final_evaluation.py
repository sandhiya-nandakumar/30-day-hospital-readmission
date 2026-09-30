import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

from imblearn.over_sampling import BorderlineSMOTE

from src.preprocessing import prepare_features, build_preprocessor, make_kernel
from src.models import build_models


DATA_PATH = "data/diabetic_data.csv"
RANDOM_STATE = 42
SAMPLE_SIZE = 20000


# =========================================================
# 1. LOAD DATA
# =========================================================

print("=" * 70)
print("FINAL MODEL COMPARISON + ERROR ANALYSIS")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print("\nOriginal dataset:", df.shape)


# =========================================================
# 2. SAME STRATIFIED 20,000 SAMPLE FOR EVERY MODEL
# =========================================================

df_sample, _ = train_test_split(
    df,
    train_size=SAMPLE_SIZE,
    stratify=df["readmitted"],
    random_state=RANDOM_STATE
)

df_sample = df_sample.reset_index(drop=True)

print("Final experiment dataset:", df_sample.shape)


# =========================================================
# 3. FEATURES + TARGET
# =========================================================

X, y = prepare_features(df_sample)

print("\nFeature shape:", X.shape)

print("\nTarget distribution:")
print(y.value_counts())


# =========================================================
# 4. SAME TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================================================
# 5. PREPROCESS
# =========================================================

preprocessor = build_preprocessor()

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(
    "\nPreprocessed training shape:",
    X_train_processed.shape
)


# =========================================================
# 6. BASELINE LOGISTIC REGRESSION
# =========================================================

from sklearn.linear_model import LogisticRegression

smote = BorderlineSMOTE(
    random_state=RANDOM_STATE
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_processed,
    y_train
)

print(
    "After SMOTE:",
    X_train_smote.shape
)


baseline_lr = LogisticRegression(
    C=0.01,
    solver="lbfgs",
    class_weight=None,
    max_iter=1000,
    random_state=RANDOM_STATE
)

baseline_lr.fit(
    X_train_smote,
    y_train_smote
)


# =========================================================
# 7. ADVANCED MODELS
# =========================================================

models = build_models()

trained_models = {}

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(
        X_train_smote,
        y_train_smote
    )

    trained_models[name] = model


# =========================================================
# 8. EVALUATION FUNCTION
# =========================================================

def evaluate_model(name, model):

    predictions = model.predict(X_test_processed)

    probabilities = model.predict_proba(
        X_test_processed
    )[:, 1]

    cm = confusion_matrix(
        y_test,
        predictions
    )

    results = {
        "model": name,
        "accuracy": accuracy_score(
            y_test,
            predictions
        ),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities
        ),
        "pr_auc": average_precision_score(
            y_test,
            probabilities
        ),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1])
    }

    return results, predictions, probabilities


# =========================================================
# 9. EVALUATE ALL MODELS
# =========================================================

all_results = []

predictions_dict = {}
probabilities_dict = {}

result, pred, prob = evaluate_model(
    "Logistic Regression",
    baseline_lr
)

all_results.append(result)

predictions_dict["Logistic Regression"] = pred
probabilities_dict["Logistic Regression"] = prob


for name, model in trained_models.items():

    result, pred, prob = evaluate_model(
        name,
        model
    )

    all_results.append(result)

    predictions_dict[name] = pred
    probabilities_dict[name] = prob


# =========================================================
# 10. COMPARISON TABLE
# =========================================================

results_df = pd.DataFrame(all_results)

print("\n")
print("=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    results_df[
        [
            "model",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc"
        ]
    ].to_string(index=False)
)


# =========================================================
# 11. CONFUSION MATRICES
# =========================================================

print("\n")
print("=" * 70)
print("CONFUSION MATRICES")
print("=" * 70)

for result in all_results:

    print("\n", result["model"])

    print(
        f"TN = {result['tn']}"
    )

    print(
        f"FP = {result['fp']}"
    )

    print(
        f"FN = {result['fn']}"
    )

    print(
        f"TP = {result['tp']}"
    )


# =========================================================
# 12. DETAILED ERROR ANALYSIS
# =========================================================

print("\n")
print("=" * 70)
print("ERROR ANALYSIS")
print("=" * 70)


for name in predictions_dict:

    predictions = predictions_dict[name]

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions
    ).ravel()

    false_positive_rate = fp / (fp + tn)

    false_negative_rate = fn / (fn + tp)

    print("\n----------------------------------------")
    print(name)
    print("----------------------------------------")

    print("False Positives:", fp)
    print("False Negatives:", fn)

    print(
        "False Positive Rate:",
        round(false_positive_rate, 4)
    )

    print(
        "False Negative Rate:",
        round(false_negative_rate, 4)
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "No early readmission",
                "Early readmission"
            ],
            zero_division=0
        )
    )


# =========================================================
# 13. FALSE NEGATIVE ANALYSIS
# =========================================================

print("\n")
print("=" * 70)
print("FALSE NEGATIVE ANALYSIS")
print("=" * 70)


# Use the stacking model for detailed analysis
stacking_predictions = predictions_dict[
    "Stacking_Ensemble"
]

stacking_probabilities = probabilities_dict[
    "Stacking_Ensemble"
]


error_analysis = X_test.copy()

error_analysis["actual"] = y_test.values

error_analysis["predicted"] = stacking_predictions

error_analysis["probability"] = stacking_probabilities


false_negatives = error_analysis[
    (error_analysis["actual"] == 1)
    &
    (error_analysis["predicted"] == 0)
]

false_positives = error_analysis[
    (error_analysis["actual"] == 0)
    &
    (error_analysis["predicted"] == 1)
]


print(
    "\nStacking Ensemble False Negatives:",
    len(false_negatives)
)

print(
    "Stacking Ensemble False Positives:",
    len(false_positives)
)


print("\nFalse Negative sample:")
print(
    false_negatives.head(10).to_string()
)


print("\nFalse Positive sample:")
print(
    false_positives.head(10).to_string()
)


# =========================================================
# 14. SAVE RESULTS
# =========================================================

os.makedirs(
    "evaluation_results",
    exist_ok=True
)

results_df.to_csv(
    "evaluation_results/final_model_comparison.csv",
    index=False
)

error_analysis.to_csv(
    "evaluation_results/stacking_error_analysis.csv",
    index=False
)


with open(
    "evaluation_results/final_model_results.json",
    "w"
) as f:

    json.dump(
        all_results,
        f,
        indent=4
    )


print("\n")
print("=" * 70)
print("FINAL EVALUATION COMPLETED")
print("=" * 70)

print("\nSaved:")
print(
    "evaluation_results/final_model_comparison.csv"
)

print(
    "evaluation_results/final_model_results.json"
)

print(
    "evaluation_results/stacking_error_analysis.csv"
)