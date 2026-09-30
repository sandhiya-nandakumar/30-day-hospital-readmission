# 30-Day Hospital Readmission Risk Prediction

Prototype for the Advanced Machine Learning (19MAM51) project.

## Problem
Predict whether a diabetic patient is likely to be readmitted to hospital within 30 days using clinical encounter information.

## Dataset
**Diabetes 130-US Hospitals for Years 1999-2008**

- 101,766 hospital encounters
- 47 documented features in the UCI version
- Target: `<30` days vs other readmission outcomes
- Original data source: UCI Machine Learning Repository
- Kaggle copy: https://www.kaggle.com/datasets/brandao/diabetes

The prototype downloads the Kaggle dataset automatically when `download_dataset.py` is run. You can also place `diabetic_data.csv` manually in `data/`.

## Advanced ML pipeline
1. Missing-value handling
2. Feature engineering
3. One-hot encoding + RobustScaler
4. RBF Nystroem kernel approximation
5. Borderline-SMOTE on the training split
6. RBF SVM
7. Random Forest
8. XGBoost
9. Stacking ensemble
10. Evaluation with PR-AUC, ROC-AUC, Recall, F1 and confusion matrix

The prototype uses a stratified sample during training so it can run on a normal laptop. Change `MAX_ROWS` in `train.py` if more data is available.

## Run

```bash
pip install -r requirements.txt
python download_dataset.py
python train.py
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal.

## Important
This is an educational prototype, not a clinical diagnostic tool. Predictions must not be used for real patient-care decisions.
