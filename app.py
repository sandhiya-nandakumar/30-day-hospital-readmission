import os
import json
import sqlite3
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="30-Day Hospital Readmission Risk Prediction",
    page_icon="favicon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(BASE_DIR, "models")
DB_FILE = os.path.join(BASE_DIR, "predictions.db")

PBIX_FILE = os.path.join(
    BASE_DIR,
    "Hospital_Readmission_Analytics.pbix"
)

DASHBOARD_IMAGE = os.path.join(
    BASE_DIR,
    "powerbi_dashboard_preview.png"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.5rem;
        font-weight: 650;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
    }

    .dashboard-card {
        padding: 1.2rem;
        border-radius: 15px;
        border: 1px solid #e5e7eb;
        background: #ffffff;
        min-height: 180px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    }

    .dashboard-card h3 {
        margin-top: 0;
        font-size: 1.15rem;
    }

    .dashboard-card p {
        color: #6b7280;
        line-height: 1.6;
    }

    .metric-card {
        padding: 1rem;
        border-radius: 14px;
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        text-align: center;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
    }

    .metric-label {
        color: #6b7280;
        font-size: 0.9rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD RESOURCES
# ============================================================

@st.cache_resource
def load_resources():

    required_files = {
        "preprocessor": "preprocessor.joblib",
        "kernel": "kernel.joblib",
        "RBF_SVM": "RBF_SVM.joblib",
        "Random_Forest": "Random_Forest.joblib",
        "XGBoost": "XGBoost.joblib",
        "Stacking_Ensemble": "Stacking_Ensemble.joblib"
    }

    resources = {}

    for key, filename in required_files.items():

        filepath = os.path.join(MODELS_DIR, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Required model file not found:\n{filepath}"
            )

        resources[key] = joblib.load(filepath)

    # Thresholds
    threshold_file = os.path.join(
        MODELS_DIR,
        "thresholds.json"
    )

    if os.path.exists(threshold_file):

        with open(threshold_file, "r") as f:
            resources["thresholds"] = json.load(f)

    else:

        resources["thresholds"] = {}

    return resources


# ============================================================
# DATABASE
# ============================================================

def initialize_database():

    conn = sqlite3.connect(DB_FILE)

    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            model TEXT,
            risk REAL,
            tier TEXT
        )
        """
    )

    conn.commit()
    conn.close()


def save_prediction(model, risk, tier):

    conn = sqlite3.connect(DB_FILE)

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO predictions
        (created_at, model, risk, tier)
        VALUES (?, ?, ?, ?)
        """,
        (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model,
            float(risk),
            tier
        )
    )

    conn.commit()
    conn.close()


def load_prediction_history():

    conn = sqlite3.connect(DB_FILE)

    df = pd.read_sql_query(
        "SELECT * FROM predictions ORDER BY id DESC",
        conn
    )

    conn.close()

    return df


# ============================================================
# RISK TIER
# ============================================================

def get_risk_tier(probability, threshold=0.50):

    if probability >= threshold:
        return "Elevated"

    elif probability >= threshold / 2:
        return "Moderate"

    else:
        return "Low"


# ============================================================
# LOAD EVERYTHING
# ============================================================

initialize_database()

try:

    resources = load_resources()

except Exception as e:

    st.error("Unable to load the machine-learning models.")

    st.code(str(e))

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🏥 Readmission Prediction")

st.sidebar.markdown(
    """
    ### Navigation

    Use the menu below to explore the application.
    """
)

page = st.sidebar.radio(
    "Select Module",
    [
        "Risk Prediction",
        "Model Analysis",
        "Prediction History",
        "Analytics Dashboard"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "30-Day Hospital Readmission Risk Prediction"
)

st.sidebar.caption(
    "Advanced Machine Learning Project"
)


# ============================================================
# 1. RISK PREDICTION
# ============================================================

if page == "Risk Prediction":

    st.markdown(
        '<div class="main-title">🏥 30-Day Hospital Readmission Risk Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Predict the probability of a patient being readmitted within 30 days.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    # --------------------------------------------------------
    # MODEL SELECTION
    # --------------------------------------------------------

    st.subheader("🤖 Select Machine Learning Model")

    model_display_names = {
        "RBF SVM": "RBF_SVM",
        "Random Forest": "Random_Forest",
        "XGBoost": "XGBoost",
        "Stacking Ensemble": "Stacking_Ensemble"
    }

    selected_display = st.selectbox(
        "Choose a model",
        list(model_display_names.keys())
    )

    selected_model_key = model_display_names[selected_display]

    model = resources[selected_model_key]

    # --------------------------------------------------------
    # INPUT SECTION
    # --------------------------------------------------------

    st.subheader("👤 Patient Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        race = st.selectbox(
            "Race",
            [
                "Caucasian",
                "AfricanAmerican",
                "Asian",
                "Hispanic",
                "Other",
                "Unknown"
            ]
        )

        gender = st.selectbox(
            "Gender",
            [
                "Female",
                "Male"
            ]
        )

        age = st.selectbox(
            "Age Group",
            [
                "[0-10)",
                "[10-20)",
                "[20-30)",
                "[30-40)",
                "[40-50)",
                "[50-60)",
                "[60-70)",
                "[70-80)",
                "[80-90)",
                "[90-100)"
            ]
        )

        admission_type_id = st.number_input(
            "Admission Type ID",
            min_value=1,
            max_value=8,
            value=1
        )

        discharge_disposition_id = st.number_input(
            "Discharge Disposition ID",
            min_value=1,
            max_value=30,
            value=1
        )

    with col2:

        admission_source_id = st.number_input(
            "Admission Source ID",
            min_value=1,
            max_value=25,
            value=1
        )

        time_in_hospital = st.number_input(
            "Time in Hospital (days)",
            min_value=1,
            max_value=14,
            value=3
        )

        num_lab_procedures = st.number_input(
            "Number of Lab Procedures",
            min_value=0,
            max_value=150,
            value=40
        )

        num_procedures = st.number_input(
            "Number of Procedures",
            min_value=0,
            max_value=10,
            value=1
        )

        num_medications = st.number_input(
            "Number of Medications",
            min_value=0,
            max_value=100,
            value=15
        )

    with col3:

        number_outpatient = st.number_input(
            "Previous Outpatient Visits",
            min_value=0,
            max_value=50,
            value=0
        )

        number_emergency = st.number_input(
            "Previous Emergency Visits",
            min_value=0,
            max_value=50,
            value=0
        )

        number_inpatient = st.number_input(
            "Previous Inpatient Visits",
            min_value=0,
            max_value=50,
            value=0
        )

        number_diagnoses = st.number_input(
            "Number of Diagnoses",
            min_value=1,
            max_value=20,
            value=5
        )

        service_utilization = (
            number_outpatient
            + number_emergency
            + number_inpatient
        )

        prior_inpatient_plus_emergency = (
            number_inpatient
            + number_emergency
        )

    st.markdown("---")

    st.subheader("📋 Prediction Summary")

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:

        st.metric(
            "Service Utilization",
            service_utilization
        )

    with summary_col2:

        st.metric(
            "Prior Inpatient + Emergency",
            prior_inpatient_plus_emergency
        )

    # --------------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------------

    st.markdown("")

    predict_button = st.button(
        "🔍 Predict 30-Day Readmission Risk",
        type="primary",
        width="stretch"
    )

    if predict_button:

        input_data = pd.DataFrame(
            {
                "race": [race],
                "gender": [gender],
                "age": [age],
                "admission_type_id": [admission_type_id],
                "discharge_disposition_id": [
                    discharge_disposition_id
                ],
                "admission_source_id": [
                    admission_source_id
                ],
                "time_in_hospital": [
                    time_in_hospital
                ],
                "num_lab_procedures": [
                    num_lab_procedures
                ],
                "num_procedures": [
                    num_procedures
                ],
                "num_medications": [
                    num_medications
                ],
                "number_outpatient": [
                    number_outpatient
                ],
                "number_emergency": [
                    number_emergency
                ],
                "number_inpatient": [
                    number_inpatient
                ],
                "number_diagnoses": [
                    number_diagnoses
                ],
                "service_utilization": [
                    service_utilization
                ],
                "prior_inpatient_plus_emergency": [
                    prior_inpatient_plus_emergency
                ]
            }
        )

        try:

            # Preprocess
            processed_data = resources[
                "preprocessor"
            ].transform(input_data)

            # Apply Nystroem kernel for SVM
            if selected_model_key == "RBF_SVM":

                processed_data = resources[
                    "kernel"
                ].transform(processed_data)

            # Prediction probability
            probability = model.predict_proba(
                processed_data
            )[0][1]

            # Get threshold
            threshold = resources[
                "thresholds"
            ].get(
                selected_model_key,
                0.50
            )

            try:
                threshold = float(threshold)
            except:
                threshold = 0.50

            # Risk tier
            tier = get_risk_tier(
                probability,
                threshold
            )

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            st.markdown("---")

            st.subheader("📊 Prediction Result")

            result_col1, result_col2, result_col3 = st.columns(3)

            with result_col1:

                st.metric(
                    "Readmission Probability",
                    f"{probability * 100:.2f}%"
                )

            with result_col2:

                st.metric(
                    "Decision Threshold",
                    f"{threshold:.2f}"
                )

            with result_col3:

                st.metric(
                    "Risk Level",
                    tier
                )

            # Visual message
            if tier == "Elevated":

                st.error(
                    f"⚠️ Elevated Risk — "
                    f"Predicted probability: "
                    f"{probability * 100:.2f}%"
                )

            elif tier == "Moderate":

                st.warning(
                    f"⚠️ Moderate Risk — "
                    f"Predicted probability: "
                    f"{probability * 100:.2f}%"
                )

            else:

                st.success(
                    f"✅ Low Risk — "
                    f"Predicted probability: "
                    f"{probability * 100:.2f}%"
                )

            # Save prediction
            save_prediction(
                selected_display,
                probability,
                tier
            )

            st.success(
                "Prediction saved to prediction history."
            )

        except Exception as e:

            st.error(
                "An error occurred while generating the prediction."
            )

            st.code(str(e))


# ============================================================
# 2. MODEL ANALYSIS
# ============================================================

elif page == "Model Analysis":

    st.title("🤖 Model Analysis")

    st.markdown(
        "Comparison of the machine-learning models evaluated "
        "for 30-day hospital readmission prediction."
    )

    st.markdown("---")

    # --------------------------------------------------------
    # FINAL STANDARDIZED RESULTS
    # --------------------------------------------------------

    results = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "RBF-SVM",
                "Random Forest",
                "XGBoost",
                "Stacking Ensemble"
            ],

            "Accuracy": [
                0.66250,
                0.69250,
                0.85425,
                0.88375,
                0.87325
            ],

            "Precision": [
                0.168135,
                0.154930,
                0.229249,
                0.193548,
                0.305732
            ],

            "Recall": [
                0.513453,
                0.394619,
                0.130045,
                0.013453,
                0.107623
            ],

            "F1 Score": [
                0.253319,
                0.222503,
                0.165951,
                0.025157,
                0.159204
            ],

            "ROC-AUC": [
                0.643175,
                0.590240,
                0.614916,
                0.630803,
                0.626352
            ],

            "PR-AUC": [
                0.204469,
                0.142901,
                0.181986,
                0.182773,
                0.184132
            ]
        }
    )

    st.dataframe(
        results.style.format(
            {
                "Accuracy": "{:.3f}",
                "Precision": "{:.3f}",
                "Recall": "{:.3f}",
                "F1 Score": "{:.3f}",
                "ROC-AUC": "{:.3f}",
                "PR-AUC": "{:.3f}"
            }
        ),
        width="stretch",
        hide_index=True
    )

    st.markdown("---")

    # --------------------------------------------------------
    # METRIC VISUALIZATION
    # --------------------------------------------------------

    st.subheader("📈 Model Metrics")

    metric_choice = st.selectbox(
        "Select metric",
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC",
            "PR-AUC"
        ]
    )

    chart_data = results[
        ["Model", metric_choice]
    ].set_index("Model")

    st.bar_chart(
        chart_data,
        height=400
    )

    st.markdown("---")

    # --------------------------------------------------------
    # ERROR ANALYSIS
    # --------------------------------------------------------

    st.subheader("🔎 Error Analysis")

    error_data = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "RBF-SVM",
                "Random Forest",
                "XGBoost",
                "Stacking Ensemble"
            ],

            "False Positive Rate": [
                0.3188,
                0.2701,
                0.0549,
                0.0070,
                0.0307
            ],

            "False Negative Rate": [
                0.4865,
                0.6054,
                0.8700,
                0.9865,
                0.8924
            ]
        }
    )

    st.dataframe(
        error_data.style.format(
            {
                "False Positive Rate": "{:.3f}",
                "False Negative Rate": "{:.3f}"
            }
        ),
        width="stretch",
        hide_index=True
    )

    st.info(
        "The dataset is imbalanced, with 30-day readmissions "
        "representing a minority class. Therefore, accuracy alone "
        "does not fully describe model performance. Recall, F1 score, "
        "ROC-AUC and PR-AUC are also considered."
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX DATA
    # --------------------------------------------------------

    st.subheader("🧮 Confusion Matrix Summary")

    confusion_data = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "RBF-SVM",
                "Random Forest",
                "XGBoost",
                "Stacking Ensemble"
            ],

            "True Negative": [
                2421,
                2594,
                3359,
                3529,
                3445
            ],

            "False Positive": [
                1133,
                960,
                195,
                25,
                109
            ],

            "False Negative": [
                217,
                270,
                388,
                440,
                398
            ],

            "True Positive": [
                229,
                176,
                58,
                6,
                48
            ]
        }
    )

    st.dataframe(
        confusion_data,
        width="stretch",
        hide_index=True
    )


# ============================================================
# 3. PREDICTION HISTORY
# ============================================================

elif page == "Prediction History":

    st.title("📜 Prediction History")

    st.markdown(
        "View predictions generated through the hospital "
        "readmission risk prediction application."
    )

    st.markdown("---")

    history = load_prediction_history()

    if history.empty:

        st.info(
            "No predictions have been generated yet."
        )

    else:

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Total Predictions",
                len(history)
            )

        with col2:

            st.metric(
                "Average Risk",
                f"{history['risk'].mean() * 100:.2f}%"
            )

        with col3:

            elevated_count = (
                history["tier"]
                .eq("Elevated")
                .sum()
            )

            st.metric(
                "Elevated Risk Predictions",
                elevated_count
            )

        st.markdown("---")

        # ----------------------------------------------------
        # HISTORY TABLE
        # ----------------------------------------------------

        display_history = history.copy()

        display_history["risk"] = (
            display_history["risk"] * 100
        ).round(2)

        display_history = display_history.rename(
            columns={
                "id": "ID",
                "created_at": "Date & Time",
                "model": "Model",
                "risk": "Risk Probability (%)",
                "tier": "Risk Level"
            }
        )

        st.dataframe(
            display_history,
            width="stretch",
            hide_index=True
        )

        st.markdown("---")

        # ----------------------------------------------------
        # RISK DISTRIBUTION
        # ----------------------------------------------------

        st.subheader("📊 Risk Distribution")

        risk_counts = (
            history["tier"]
            .value_counts()
            .rename_axis("Risk Level")
            .to_frame("Predictions")
        )

        st.bar_chart(
            risk_counts,
            height=300
        )


# ============================================================
# 4. ANALYTICS DASHBOARD
# ============================================================

elif page == "Analytics Dashboard":

    st.title("📊 Analytics Dashboard")

    st.markdown(
        "Explore hospital readmission patterns, risk factors, "
        "machine-learning performance, and prediction activity."
    )

    st.markdown("---")

    # --------------------------------------------------------
    # DASHBOARD HEADER
    # --------------------------------------------------------

    st.subheader("🏥 30-Day Hospital Readmission Analytics")

    st.markdown(
        """
        The Power BI dashboard provides an interactive view of
        hospital readmission patterns and the performance of the
        machine-learning models developed in this project.
        """
    )

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">
                    🏥 Hospital Encounters
                </div>
                <div class="metric-value">
                    101,766
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">
                    🔄 30-Day Readmissions
                </div>
                <div class="metric-value">
                    11,357
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">
                    📈 Readmission Rate
                </div>
                <div class="metric-value">
                    11.16%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">
                    🤖 ML Models
                </div>
                <div class="metric-value">
                    5
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # --------------------------------------------------------
    # POWER BI PREVIEW
    # --------------------------------------------------------

    st.subheader("📈 Power BI Dashboard")

    st.markdown(
        """
        The analytics dashboard contains four major views covering
        hospital statistics, readmission risk factors, machine-learning
        performance, and prediction monitoring.
        """
    )

    if os.path.exists(DASHBOARD_IMAGE):

        st.image(
            DASHBOARD_IMAGE,
            caption="30-Day Hospital Readmission Analytics",
            width="stretch"
        )

    else:

        st.info(
            "📊 Power BI dashboard preview image can be added to "
            "the project folder as "
            "`powerbi_dashboard_preview.png`."
        )

    st.markdown("---")

    # --------------------------------------------------------
    # DASHBOARD VIEWS
    # --------------------------------------------------------

    st.subheader("🔍 Dashboard Views")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class="dashboard-card">

            ### 🏥 Hospital Overview

            Provides an overview of the hospital dataset and
            readmission patterns.

            **Includes**

            • Total hospital encounters  
            • 30-day readmission count  
            • Overall readmission rate  
            • Readmission distribution  
            • Age-group analysis  
            • Hospital stay duration  
            • Previous inpatient visits  
            • Previous emergency visits  

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="dashboard-card">

            ### ⚠️ Readmission Risk Factors

            Examines variables associated with hospital
            readmission patterns.

            **Includes**

            • Previous emergency visits  
            • Number of medications  
            • Admission type  
            • Service utilization  
            • Hospital stay duration  
            • Previous inpatient visits  

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("")

    col3, col4 = st.columns(2)

    with col3:

        st.markdown(
            """
            <div class="dashboard-card">

            ### 🤖 ML Model Performance

            Compares the machine-learning approaches used
            in the project.

            **Models**

            • Logistic Regression  
            • RBF-SVM  
            • Random Forest  
            • XGBoost  
            • Stacking Ensemble  

            **Metrics**

            • Accuracy  
            • Precision  
            • Recall  
            • F1 Score  
            • ROC-AUC  
            • PR-AUC  

            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:

        st.markdown(
            """
            <div class="dashboard-card">

            ### 📊 Prediction Monitoring

            Monitors predictions generated by the Streamlit
            application.

            **Includes**

            • Total predictions  
            • Risk-level distribution  
            • Predictions by ML model  
            • Risk probability over time  

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # --------------------------------------------------------
    # POWER BI REPORT
    # --------------------------------------------------------

    st.subheader("🖥️ Interactive Power BI Report")

    st.markdown(
        """
        The complete interactive analytics report was developed
        using **Microsoft Power BI Desktop**.

        It contains the following pages:

        **Hospital Overview**  
        **Readmission Risk Factors**  
        **ML Model Performance**  
        **Prediction Monitoring**
        """
    )

    if os.path.exists(PBIX_FILE):

        st.success(
            "✅ Interactive Power BI report is available in the project."
        )

        st.markdown(
            """
            Open the **Hospital_Readmission_Analytics.pbix** file
            using Microsoft Power BI Desktop to explore the complete
            interactive dashboard.
            """
        )

    else:

        st.warning(
            "Power BI report file is not currently available "
            "in the project directory."
        )

    st.markdown("---")

    st.caption(
        "Power BI is used for interactive analytics and visualization "
        "of the hospital readmission dataset and prediction history."
    )