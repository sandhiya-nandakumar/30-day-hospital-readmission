from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import BorderlineSMOTE

from src.preprocessing import (
    prepare_features,
    build_preprocessor,
    make_kernel
)

from src.models import build_models
from src.metrics import calculate_metrics


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "diabetic_data.csv"
MODELS_DIR = BASE_DIR / "models"

MODELS_DIR.mkdir(exist_ok=True)

MAX_ROWS = 20000
RANDOM_STATE = 42


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 60)
print("30-DAY HOSPITAL READMISSION PREDICTION")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Full dataset shape: {df.shape}")


# ============================================================
# 2. STRATIFIED SAMPLING
# ============================================================

if len(df) > MAX_ROWS:

    print(f"\nCreating stratified sample of {MAX_ROWS} rows...")

    df, _ = train_test_split(
        df,
        train_size=MAX_ROWS,
        random_state=RANDOM_STATE,
        stratify=df["readmitted"]
    )

print(f"Working dataset shape: {df.shape}")


# ============================================================
# 3. PREPARE FEATURES AND TARGET
# ============================================================

print("\nPreparing features...")

X, y = prepare_features(df)

print(f"Feature shape: {X.shape}")
print(f"Positive cases (<30 days): {y.sum()}")
print(f"Negative cases: {(y == 0).sum()}")


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ============================================================
# 5. PREPROCESSING
# ============================================================

print("\nBuilding preprocessing pipeline...")

preprocessor = build_preprocessor()

X_train_p = preprocessor.fit_transform(X_train)
X_test_p = preprocessor.transform(X_test)

print(f"Preprocessed training shape: {X_train_p.shape}")
print(f"Preprocessed testing shape: {X_test_p.shape}")


# Save preprocessor
joblib.dump(
    preprocessor,
    MODELS_DIR / "preprocessor.joblib"
)


# ============================================================
# 6. LOGISTIC REGRESSION BASELINE
# ============================================================

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION BASELINE")
print("=" * 60)

print("\nApplying Borderline-SMOTE to baseline training data...")

smote_baseline = BorderlineSMOTE(
    random_state=RANDOM_STATE,
    k_neighbors=5
)

X_train_baseline, y_train_baseline = smote_baseline.fit_resample(
    X_train_p,
    y_train
)

print(
    f"After SMOTE: {X_train_baseline.shape[0]} training samples"
)


print("\nTraining Logistic Regression...")

baseline = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=RANDOM_STATE
)

baseline.fit(
    X_train_baseline,
    y_train_baseline
)


# Predictions
baseline_prob = baseline.predict_proba(X_test_p)[:, 1]

baseline_pred = (
    baseline_prob >= 0.5
).astype(int)


# Metrics
baseline_metrics = calculate_metrics(
    y_test,
    baseline_pred,
    baseline_prob
)

print("\nLogistic Regression Results:")

for metric, value in baseline_metrics.items():
    print(f"{metric}: {value}")


# Save baseline
joblib.dump(
    baseline,
    MODELS_DIR / "Logistic_Regression.joblib"
)


# ============================================================
# 7. RBF NYSTROEM KERNEL
# ============================================================

print("\n" + "=" * 60)
print("RBF KERNEL TRANSFORMATION")
print("=" * 60)

print("\nCreating RBF Nystroem kernel...")

kernel = make_kernel(
    n_components=40,
    gamma=0.1
)

X_train_k = kernel.fit_transform(X_train_p)
X_test_k = kernel.transform(X_test_p)

print(f"Kernel training shape: {X_train_k.shape}")
print(f"Kernel testing shape: {X_test_k.shape}")


# Save kernel
joblib.dump(
    kernel,
    MODELS_DIR / "kernel.joblib"
)


# ============================================================
# 8. SMOTE FOR ADVANCED MODELS
# ============================================================

print("\nApplying Borderline-SMOTE to advanced models...")

smote = BorderlineSMOTE(
    random_state=RANDOM_STATE,
    k_neighbors=5
)

X_train_r, y_train_r = smote.fit_resample(
    X_train_k,
    y_train
)

print(
    f"Advanced-model training samples after SMOTE: "
    f"{X_train_r.shape[0]}"
)


# ============================================================
# 9. BUILD ADVANCED MODELS
# ============================================================

print("\n" + "=" * 60)
print("BUILDING ADVANCED MODELS")
print("=" * 60)

models = build_models()


# ============================================================
# 10. INITIALIZE RESULTS
# ============================================================

results = {
    "Logistic_Regression": baseline_metrics
}


# ============================================================
# 11. TRAIN ADVANCED MODELS
# ============================================================

for name, model in models.items():

    print("\n" + "-" * 60)
    print(f"Training {name}...")
    print("-" * 60)

    model.fit(
        X_train_r,
        y_train_r
    )

    y_prob = model.predict_proba(X_test_k)[:, 1]

    y_pred = (
        y_prob >= 0.5
    ).astype(int)

    metrics = calculate_metrics(
        y_test,
        y_pred,
        y_prob
    )

    results[name] = metrics

    print(f"\n{name} Results:")

    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    filename = f"{name}.joblib"

    joblib.dump(
        model,
        MODELS_DIR / filename
    )

    print(f"\nSaved: models/{filename}")


# ============================================================
# 12. SAVE METRICS
# ============================================================

metrics_path = MODELS_DIR / "metrics.json"

with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2
    )


# ============================================================
# 13. FINAL MODEL COMPARISON
# ============================================================

print("\n" + "=" * 90)
print("FINAL MODEL COMPARISON")
print("=" * 90)

print(
    f"{'Model':<25}"
    f"{'Accuracy':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
    f"{'ROC-AUC':<12}"
    f"{'PR-AUC':<12}"
)

print("-" * 90)

for name, metrics in results.items():

    print(
        f"{name:<25}"
        f"{metrics['accuracy']:<12.4f}"
        f"{metrics['precision']:<12.4f}"
        f"{metrics['recall']:<12.4f}"
        f"{metrics['f1']:<12.4f}"
        f"{metrics['roc_auc']:<12.4f}"
        f"{metrics['pr_auc']:<12.4f}"
    )


# ============================================================
# 14. FINISHED
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nMetrics saved to:")
print(metrics_path)

print("\nModels saved in:")
print(MODELS_DIR)