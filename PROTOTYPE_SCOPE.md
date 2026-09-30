# Prototype Scope

## Project title
30-Day Hospital Readmission Risk Prediction

## Real-world problem
Hospitals need to identify patients who may be readmitted within 30 days so that high-risk cases can receive additional review and follow-up planning.

## User flow
1. User opens the Streamlit dashboard.
2. User enters patient admission and utilization information.
3. User selects an ML model.
4. Backend preprocesses the input using the saved training pipeline.
5. The input is mapped into an RBF kernel feature space.
6. The selected model returns a probability.
7. The UI displays the probability and Low/Moderate/High tier.
8. The prediction is stored in SQLite history.

## Advanced ML components
- RBF Support Vector Machine
- RBF kernel approximation with Nystroem
- Borderline-SMOTE
- Random Forest bagging
- XGBoost boosting
- Stacking ensemble with logistic-regression meta learner
- PR-AUC, ROC-AUC, Recall and F1 evaluation

## Dataset
Kaggle: https://www.kaggle.com/datasets/brandao/diabetes
Original benchmark: UCI Diabetes 130-US Hospitals for Years 1999-2008.

## Prototype limitation
The training script samples 20,000 encounters by default for laptop-friendly execution. Increase MAX_ROWS after confirming that your machine can handle the workload.
