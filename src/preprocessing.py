import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.kernel_approximation import Nystroem


# ============================================================
# FEATURE CONFIGURATION
# ============================================================

NUMERIC_FEATURES = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "service_utilization",
    "prior_inpatient_plus_emergency"
]

CATEGORICAL_FEATURES = [
    "race",
    "gender",
    "age"
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data(path):

    df = pd.read_csv(path)

    return df


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df):

    df = df.copy()

    # --------------------------------------------------------
    # Convert unknown values represented by ?
    # --------------------------------------------------------

    df = df.replace("?", "Unknown")

    # --------------------------------------------------------
    # Create target
    #
    # 1 = readmitted within 30 days
    # 0 = not readmitted within 30 days
    # --------------------------------------------------------

    y = (
        df["readmitted"]
        .astype(str)
        .eq("<30")
        .astype(int)
    )

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    df["service_utilization"] = (
        df["number_outpatient"]
        + df["number_emergency"]
        + df["number_inpatient"]
    )

    df["prior_inpatient_plus_emergency"] = (
        df["number_inpatient"]
        + df["number_emergency"]
    )

    # --------------------------------------------------------
    # Select features
    # --------------------------------------------------------

    selected_features = [
        "race",
        "gender",
        "age",
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
        "service_utilization",
        "prior_inpatient_plus_emergency"
    ]

    X = df[selected_features].copy()

    return X, y


# ============================================================
# BUILD PREPROCESSOR
# ============================================================

def build_preprocessor():

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                RobustScaler(),
                NUMERIC_FEATURES
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                CATEGORICAL_FEATURES
            )
        ]
    )

    return preprocessor


# ============================================================
# RBF NYSTROEM KERNEL
# ============================================================

def make_kernel(
    n_components=40,
    gamma=0.1
):

    kernel = Nystroem(
        kernel="rbf",
        gamma=gamma,
        n_components=n_components,
        random_state=42
    )

    return kernel