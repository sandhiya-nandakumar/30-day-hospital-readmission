import pandas as pd
import os

# Input dataset
input_file = "data/diabetic_data.csv"

# Output file
output_file = "powerbi_readmission_data.csv"

# Load dataset
df = pd.read_csv(input_file)

# Replace missing markers
df = df.replace("?", "Unknown")

# Create 30-day readmission target
df["readmission_30_day"] = (
    df["readmitted"].astype(str).eq("<30").astype(int)
)

# Create readable readmission label
df["readmission_status"] = df["readmission_30_day"].map({
    0: "Not Readmitted Within 30 Days",
    1: "Readmitted Within 30 Days"
})

# Convert age ranges into readable groups
df["age_group"] = df["age"].astype(str)

# Hospital stay groups
df["hospital_stay_group"] = pd.cut(
    df["time_in_hospital"],
    bins=[0, 3, 7, 14, 100],
    labels=[
        "1-3 Days",
        "4-7 Days",
        "8-14 Days",
        "15+ Days"
    ]
)

# Service utilization
df["service_utilization"] = (
    df["number_outpatient"]
    + df["number_emergency"]
    + df["number_inpatient"]
)

# Previous inpatient + emergency visits
df["prior_inpatient_plus_emergency"] = (
    df["number_inpatient"]
    + df["number_emergency"]
)

# Select useful columns for Power BI
powerbi_columns = [
    "encounter_id",
    "patient_nbr",
    "race",
    "gender",
    "age_group",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "hospital_stay_group",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "service_utilization",
    "prior_inpatient_plus_emergency",
    "readmitted",
    "readmission_30_day",
    "readmission_status"
]

# Keep only columns that exist
powerbi_columns = [
    col for col in powerbi_columns
    if col in df.columns
]

powerbi_df = df[powerbi_columns].copy()

# Save
powerbi_df.to_csv(output_file, index=False)

print("Power BI dataset created successfully!")
print(f"File: {os.path.abspath(output_file)}")
print(f"Rows: {len(powerbi_df)}")
print(f"Columns: {len(powerbi_df.columns)}")
print("\nReadmission distribution:")
print(powerbi_df["readmission_status"].value_counts())