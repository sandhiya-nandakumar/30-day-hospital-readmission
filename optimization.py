import json
import os

import joblib
import pandas as pd

from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from src.preprocessing import prepare_features, build_preprocessor


# =========================================================
# CONFIGURATION
# =========================================================

DATA_PATH = "data/diabetic_data.csv"

SAMPLE_SIZE = 20000
TEST_SIZE = 0.20

RANDOM_STATE = 42

MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


# =========================================================
# 1. LOAD DATASET
# =========================================================

print("=" * 70)
print("FINAL OPTIMIZATION")
print("=" * 70)

full_df = pd.read_csv(DATA_PATH)

print("\nOriginal dataset:")
print(full_df.shape)


# =========================================================
# 2. EXACT SAME 20,000-ROW STRATIFIED SAMPLE
# =========================================================

df, _ = train_test_split(
    full_df,
    train_size=SAMPLE_SIZE,
    stratify=full_df["readmitted"],
    random_state=RANDOM_STATE
)

df = df.reset_index(drop=True)

print("\nFinal experiment dataset:")
print(df.shape)


# =========================================================
# 3. PREPARE FEATURES
# =========================================================

X, y = prepare_features(df)

print("\nFeature shape:", X.shape)

print("\nTarget distribution:")
print(y.value_counts())

print("\nPositive cases:", int(y.sum()))
print("Negative cases:", int(len(y) - y.sum()))


# =========================================================
# 4. FINAL TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================================================
# 5. CROSS VALIDATION
# =========================================================

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=RANDOM_STATE
)


# =========================================================
# 6. LOGISTIC REGRESSION
# =========================================================

lr_pipeline = Pipeline(
    steps=[
        ("preprocess", build_preprocessor()),

        (
            "smote",
            BorderlineSMOTE(
                random_state=RANDOM_STATE
            )
        ),

        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE
            )
        )
    ]
)


lr_params = {
    "model__C": [0.01, 0.1, 1, 10, 100],
    "model__solver": ["liblinear", "lbfgs"],
    "model__class_weight": [None, "balanced"]
}


lr_search = RandomizedSearchCV(
    estimator=lr_pipeline,
    param_distributions=lr_params,
    n_iter=6,
    scoring="f1",
    cv=cv,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1
)

print("\n" + "=" * 70)
print("OPTIMIZING LOGISTIC REGRESSION")
print("=" * 70)

lr_search.fit(X_train, y_train)

best_lr = lr_search.best_estimator_

print("\nBest Logistic Regression parameters:")
print(lr_search.best_params_)

print("\nBest CV F1:")
print(lr_search.best_score_)


# =========================================================
# 7. RANDOM FOREST
# =========================================================

rf_pipeline = Pipeline(
    steps=[
        ("preprocess", build_preprocessor()),

        (
            "smote",
            BorderlineSMOTE(
                random_state=RANDOM_STATE
            )
        ),

        (
            "model",
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]
)


rf_params = {
    "model__n_estimators": [100, 150, 200],
    "model__max_depth": [8, 12, 16, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2"],
    "model__class_weight": [None, "balanced"]
}


rf_search = RandomizedSearchCV(
    estimator=rf_pipeline,
    param_distributions=rf_params,
    n_iter=8,
    scoring="f1",
    cv=cv,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1
)

print("\n" + "=" * 70)
print("OPTIMIZING RANDOM FOREST")
print("=" * 70)

rf_search.fit(X_train, y_train)

best_rf = rf_search.best_estimator_

print("\nBest Random Forest parameters:")
print(rf_search.best_params_)

print("\nBest CV F1:")
print(rf_search.best_score_)


# =========================================================
# 8. EVALUATION FUNCTION
# =========================================================

def evaluate_model(model, X_data, y_data, threshold=0.50):

    probabilities = model.predict_proba(X_data)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y_data,
        predictions
    )

    tn, fp, fn, tp = cm.ravel()

    return {
        "threshold": float(threshold),

        "accuracy": float(
            accuracy_score(y_data, predictions)
        ),

        "precision": float(
            precision_score(
                y_data,
                predictions,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_data,
                predictions,
                zero_division=0
            )
        ),

        "f1": float(
            f1_score(
                y_data,
                predictions,
                zero_division=0
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                y_data,
                probabilities
            )
        ),

        "pr_auc": float(
            average_precision_score(
                y_data,
                probabilities
            )
        ),

        "confusion_matrix": cm.tolist(),

        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp)
    }


# =========================================================
# 9. FINAL TEST EVALUATION
# =========================================================

print("\n" + "=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

lr_results = evaluate_model(
    best_lr,
    X_test,
    y_test,
    threshold=0.50
)

rf_results = evaluate_model(
    best_rf,
    X_test,
    y_test,
    threshold=0.50
)


print("\nOptimized Logistic Regression:")
for key, value in lr_results.items():
    print(f"{key}: {value}")


print("\nOptimized Random Forest:")
for key, value in rf_results.items():
    print(f"{key}: {value}")


# =========================================================
# 10. SAVE OPTIMIZED MODELS
# =========================================================

joblib.dump(
    best_lr,
    os.path.join(
        MODEL_DIR,
        "optimized_logistic_regression_pipeline.pkl"
    )
)

joblib.dump(
    best_rf,
    os.path.join(
        MODEL_DIR,
        "optimized_random_forest_pipeline.pkl"
    )
)


# =========================================================
# 11. SAVE OPTIMIZATION RESULTS
# =========================================================

results = {

    "experiment": {
        "dataset_size": int(len(df)),
        "training_size": int(len(X_train)),
        "testing_size": int(len(X_test)),
        "positive_cases": int(y.sum()),
        "negative_cases": int(len(y) - y.sum()),
        "random_state": RANDOM_STATE
    },

    "Logistic_Regression": {
        "best_parameters": lr_search.best_params_,
        "best_cv_f1": float(
            lr_search.best_score_
        ),
        "test_metrics": lr_results
    },

    "Random_Forest": {
        "best_parameters": rf_search.best_params_,
        "best_cv_f1": float(
            rf_search.best_score_
        ),
        "test_metrics": rf_results
    }
}


with open(
    os.path.join(
        MODEL_DIR,
        "optimization_results.json"
    ),
    "w"
) as file:

    json.dump(
        results,
        file,
        indent=4
    )


# =========================================================
# 12. FINISHED
# =========================================================

print("\n" + "=" * 70)
print("OPTIMIZATION COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nSaved files:")

print(
    "models/optimized_logistic_regression_pipeline.pkl"
)

print(
    "models/optimized_random_forest_pipeline.pkl"
)

print(
    "models/optimization_results.json"
)