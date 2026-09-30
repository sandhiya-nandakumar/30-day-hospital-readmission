from pathlib import Path
import json
import sqlite3
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_DIR = Path("models")
DB = Path("predictions.db")

st.set_page_config(
    page_title="Readmission Risk Prediction",
    page_icon="🏥",
    layout="wide"
)

def db_init():
    con = sqlite3.connect(DB)
    con.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            model TEXT,
            risk REAL,
            tier TEXT
        )
    """)
    con.commit()
    con.close()

def save_prediction(model, risk, tier):
    con = sqlite3.connect(DB)
    con.execute(
        "INSERT INTO predictions(created_at, model, risk, tier) VALUES (?, ?, ?, ?)",
        (datetime.now().isoformat(timespec="seconds"), model, float(risk), tier)
    )
    con.commit()
    con.close()

def load_history():
    con = sqlite3.connect(DB)
    df = pd.read_sql_query(
        "SELECT created_at, model, risk, tier FROM predictions ORDER BY id DESC LIMIT 20",
        con
    )
    con.close()
    return df

db_init()

st.title("🏥 30-Day Hospital Readmission Risk Prediction")

if not (MODEL_DIR / "Stacking_Ensemble.joblib").exists():
    st.warning("Model files are not available yet. Run `python download_dataset.py` and then `python train.py`.")
    st.stop()

preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")
kernel = joblib.load(MODEL_DIR / "kernel.joblib")

model_files = {
    "RBF SVM": "RBF_SVM.joblib",
    "Random Forest": "Random_Forest.joblib",
    "XGBoost": "XGBoost.joblib",
    "Stacking Ensemble": "Stacking_Ensemble.joblib"
}

with st.sidebar:
    st.header("Prediction Settings")
    selected_model = st.selectbox("Model", list(model_files))

c1, c2, c3 = st.columns(3)

with c1:
    age = st.selectbox(
        "Age Group",
        ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
         "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"],
        index=6
    )
    gender = st.selectbox("Gender", ["Female", "Male", "Unknown/Invalid"])
    race = st.selectbox(
        "Race",
        ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other", "Missing"]
    )

with c2:
    time_in_hospital = st.slider("Time in Hospital (days)", 1, 14, 4)
    num_lab_procedures = st.slider("Number of Lab Procedures", 1, 132, 40)
    num_procedures = st.slider("Number of Procedures", 0, 6, 1)
    num_medications = st.slider("Number of Medications", 1, 81, 15)

with c3:
    number_outpatient = st.slider("Outpatient Visits in Previous Year", 0, 42, 0)
    number_emergency = st.slider("Emergency Visits in Previous Year", 0, 76, 0)
    number_inpatient = st.slider("Inpatient Visits in Previous Year", 0, 21, 0)
    number_diagnoses = st.slider("Number of Diagnoses", 1, 16, 6)

admission_type_id = st.selectbox("Admission Type ID", ["1", "2", "3", "4", "5", "6", "7", "8"])
discharge_disposition_id = st.selectbox("Discharge Disposition ID", ["1", "2", "3", "6", "11", "18", "25"])
admission_source_id = st.selectbox("Admission Source ID", ["1", "2", "4", "5", "7", "9", "11", "13", "17", "20"])

if st.button("🔍 Predict Readmission Risk", type="primary"):
    row = pd.DataFrame([{
        "race": race,
        "gender": gender,
        "age": age,
        "admission_type_id": admission_type_id,
        "discharge_disposition_id": discharge_disposition_id,
        "admission_source_id": admission_source_id,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "number_diagnoses": number_diagnoses,
        "service_utilization": number_outpatient + number_emergency + number_inpatient,
        "prior_inpatient_plus_emergency": number_inpatient + number_emergency
    }])

    processed = preprocessor.transform(row)
    kernel_data = kernel.transform(processed)

    model = joblib.load(MODEL_DIR / model_files[selected_model])
    risk = float(model.predict_proba(kernel_data)[0, 1])

    if risk >= 0.50:
        tier = "High"
    elif risk >= 0.25:
        tier = "Moderate"
    else:
        tier = "Low"

    st.divider()
    r1, r2 = st.columns(2)
    with r1:
        st.metric("Predicted probability", f"{risk * 100:.1f}%")
    with r2:
        st.metric("Risk tier", tier)

    if tier == "High":
        st.error("High predicted risk — review the case using appropriate clinical protocols.")
    elif tier == "Moderate":
        st.warning("Moderate predicted risk — additional review may be appropriate.")
    else:
        st.success("Low predicted risk according to this prototype.")

    save_prediction(selected_model, risk, tier)

st.divider()
st.subheader("Recent Prediction History")
history = load_history()
if len(history):
    st.dataframe(history, use_container_width=True)
else:
    st.info("No predictions recorded yet.")
