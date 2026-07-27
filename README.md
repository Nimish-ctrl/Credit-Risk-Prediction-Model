# Loan Default Prediction (XGBoost)
 
Predicts loan default risk using the `Loan_Default.csv` dataset (148,670 records, 34 features).
 
## Pipeline
 
1. **Load & Inspect** — shape, missing values, target distribution.
2. **EDA** — 5 quick plots (class imbalance, loan amount, key relationships to `Status`).
3. **Leakage Check** — dropped 13 post-approval fields (e.g. `rate_of_interest`, `LTV`, `Credit_Worthiness`, `credit_type`) that wouldn't be known at application time, plus `ID`.
4. **Cleaning** — median imputation (numeric), mode imputation (categorical).
5. **Encoding** — one-hot encoding (`drop_first=True`).
6. **Feature Selection** — Random Forest importance → top 25 features.
7. **Train/Val Split** — 80/20, stratified.
8. **Imbalance Handling** — two approaches compared: `scale_pos_weight` vs. SMOTE.
9. **Model Tuning** — XGBoost + `RandomizedSearchCV` (Stratified 3-fold CV) for both approaches.
10. **Evaluation** — threshold tuning, confusion matrix, precision/recall/F1, ROC-AUC for both models.
11. **Interpretability** — SHAP (bar summary + beeswarm) on the higher-AUC model.
## Tech Stack
`pandas`, `scikit-learn`, `xgboost`, `imbalanced-learn`, `shap`, `matplotlib`, `seaborn`
 
## Results
 
Two imbalance-handling approaches were tuned and compared on the validation set (29,734 records, 24.65% default rate):
 
| Model | AUC-ROC | Threshold | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **A — scale_pos_weight** | **0.8415** | 0.60 | 0.7847 | 0.5805 | 0.6673 |
| B — SMOTE | 0.8076 | 0.56 | 0.7666 | 0.5243 | 0.6227 |
 
**Model A (scale_pos_weight) won on every metric** and was selected as the final model.
 
- Accuracy: 86.0%
- Best CV F1 (tuning stage): 0.6427 (Model A) vs. 0.8536 (Model B) — Model B's high CV F1 came from being tuned on SMOTE-resampled folds, which don't reflect real-world class balance; validation-set performance on the original distribution is the metric that matters, and there Model A is stronger.
- Top predictors : `property_value`, `income`, `loan_amount`, `Credit_Score` — these four alone account for the large majority of predictive signal, with the remaining 21 features contributing smaller, more incremental gains.
- Final model interpreted with SHAP (bar summary + beeswarm) to confirm feature importance direction and surface individual prediction drivers.
## How to Run
Open `loan-default-final-model.ipynb` and run all cells top to bottom. Update the CSV path in Step 1 if needed.
