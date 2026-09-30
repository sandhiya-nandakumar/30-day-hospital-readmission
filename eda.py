from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


DATA = Path("data/diabetic_data.csv")
EDA_DIR = Path("eda_results")
EDA_DIR.mkdir(exist_ok=True)


def main():

    # --------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------
    print("Loading dataset...")

    df = pd.read_csv(DATA)

    print("\nDataset shape:")
    print(df.shape)

    # --------------------------------------------------
    # 2. Basic information
    # --------------------------------------------------
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    # --------------------------------------------------
    # 3. Missing / unknown values
    # --------------------------------------------------
    print("\nMissing values:")

    missing = df.isnull().sum()
    print(missing[missing > 0])

    # In this dataset, '?' is also used for unknown values
    question_marks = (df == "?").sum()

    print("\nUnknown values represented by '?':")
    print(question_marks[question_marks > 0])

    # --------------------------------------------------
    # 4. Target distribution
    # --------------------------------------------------
    print("\nReadmission distribution:")

    print(df["readmitted"].value_counts())

    print("\nReadmission percentages:")

    print(
        df["readmitted"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    # --------------------------------------------------
    # 5. Target visualization
    # --------------------------------------------------
    plt.figure(figsize=(7, 5))

    sns.countplot(
        data=df,
        x="readmitted"
    )

    plt.title("Hospital Readmission Distribution")
    plt.xlabel("Readmission Status")
    plt.ylabel("Number of Encounters")

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "01_readmission_distribution.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------
    # 6. Age distribution
    # --------------------------------------------------
    plt.figure(figsize=(9, 5))

    sns.countplot(
        data=df,
        x="age",
        order=df["age"].value_counts().index
    )

    plt.title("Age Group Distribution")
    plt.xlabel("Age Group")
    plt.ylabel("Number of Encounters")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "02_age_distribution.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------
    # 7. Gender distribution
    # --------------------------------------------------
    plt.figure(figsize=(7, 5))

    sns.countplot(
        data=df,
        x="gender"
    )

    plt.title("Gender Distribution")
    plt.xlabel("Gender")
    plt.ylabel("Number of Encounters")

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "03_gender_distribution.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------
    # 8. Hospital stay distribution
    # --------------------------------------------------
    plt.figure(figsize=(9, 5))

    sns.histplot(
        data=df,
        x="time_in_hospital",
        bins=14
    )

    plt.title("Distribution of Hospital Stay Duration")
    plt.xlabel("Days in Hospital")
    plt.ylabel("Number of Encounters")

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "04_hospital_stay_distribution.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------
    # 9. Number of medications
    # --------------------------------------------------
    plt.figure(figsize=(9, 5))

    sns.histplot(
        data=df,
        x="num_medications",
        bins=30
    )

    plt.title("Distribution of Number of Medications")
    plt.xlabel("Number of Medications")
    plt.ylabel("Number of Encounters")

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "05_medications_distribution.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------
    # 10. Number of diagnoses
    # --------------------------------------------------
    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="number_diagnoses"
    )

    plt.title("Number of Diagnoses per Encounter")
    plt.xlabel("Number of Diagnoses")
    plt.ylabel("Number of Encounters")

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "06_diagnoses_distribution.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------
    # 11. Readmission vs hospital stay
    # --------------------------------------------------
    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="readmitted",
        y="time_in_hospital"
    )

    plt.title("Hospital Stay vs Readmission Status")
    plt.xlabel("Readmission Status")
    plt.ylabel("Days in Hospital")

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "07_stay_vs_readmission.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------
    # 12. Readmission vs inpatient visits
    # --------------------------------------------------
    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="readmitted",
        y="number_inpatient"
    )

    plt.title("Previous Inpatient Visits vs Readmission")
    plt.xlabel("Readmission Status")
    plt.ylabel("Number of Previous Inpatient Visits")

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "08_inpatient_vs_readmission.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------
    # 13. Readmission vs emergency visits
    # --------------------------------------------------
    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="readmitted",
        y="number_emergency"
    )

    plt.title("Previous Emergency Visits vs Readmission")
    plt.xlabel("Readmission Status")
    plt.ylabel("Number of Previous Emergency Visits")

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "09_emergency_vs_readmission.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------
    # 14. Save summary
    # --------------------------------------------------
    with open(EDA_DIR / "eda_summary.txt", "w") as f:

        f.write("30-Day Hospital Readmission Dataset - EDA\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Dataset shape: {df.shape}\n\n")

        f.write("Readmission distribution:\n")
        f.write(
            str(df["readmitted"].value_counts())
        )

        f.write("\n\nReadmission percentages:\n")
        f.write(
            str(
                df["readmitted"]
                .value_counts(normalize=True)
                .mul(100)
                .round(2)
            )
        )

        f.write("\n\nMissing values:\n")
        f.write(str(missing[missing > 0]))

        f.write("\n\nUnknown '?' values:\n")
        f.write(str(question_marks[question_marks > 0]))

    print("\nEDA completed successfully.")

    print(f"\nEDA results saved in:")
    print(EDA_DIR)


if __name__ == "__main__":
    main()