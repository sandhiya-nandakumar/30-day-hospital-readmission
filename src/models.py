from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def build_models():

    # 1. RBF Support Vector Machine
    svc = SVC(
        C=2.0,
        kernel="rbf",
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=42
    )

    # 2. Random Forest
    rf = RandomForestClassifier(
        n_estimators=180,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    # 3. XGBoost
    xgb = XGBClassifier(
        n_estimators=180,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=4
    )

    # 4. Stacking Ensemble
    stacking = StackingClassifier(
        estimators=[
            ("rbf_svc", svc),
            ("random_forest", rf),
            ("xgboost", xgb)
        ],
        final_estimator=LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),
        cv=3,
        n_jobs=-1,
        stack_method="predict_proba"
    )

    return {
        "RBF_SVM": svc,
        "Random_Forest": rf,
        "XGBoost": xgb,
        "Stacking_Ensemble": stacking
    }