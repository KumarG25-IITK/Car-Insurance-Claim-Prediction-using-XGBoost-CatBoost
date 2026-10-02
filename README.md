# Car Insurance Claim Probability & Severity Pipeline

## Business Objective
Predicting insurance claims is a critical two-part challenge for risk management. This project implements a two-stage machine learning pipeline to predict **whether** a policyholder will file a claim (classification) and, if so, **how much** the claim will cost (regression). This allows for dynamic premium pricing and accurate financial reserve planning.

## Architecture & Methodology
* **Data Preprocessing:** Implemented a leak-free Scikit-Learn `ColumnTransformer` pipeline. Handled missing data via `KNNImputer`, applied targeted encoding (Ordinal/One-Hot), and eliminated perfect multicollinearity (Dummy Variable Trap) using Variance Inflation Factor (VIF) analysis.
* **Stage 1 - Claim Probability (Classification):** Evaluated 10 ensemble and baseline models. Selected and tuned an **XGBoost Classifier** using the histogram tree method (`tree_method='hist'`) for high-efficiency training.
* **Stage 2 - Claim Severity (Regression):** Built an **SGDRegressor** to predict continuous claim amounts for positive-claim instances, utilizing standard K-Fold cross-validation to prevent stratification errors on continuous targets.
* **Hyperparameter Optimization:** Utilized an industry-standard two-step tuning strategy to balance computational cost and marginal gains:
    1. **Broad Sweep:** `RandomizedSearchCV` (100 iterations) to efficiently locate optimal hyperparameter neighborhoods.
    2. **Fine-Tuning:** `GridSearchCV` to exhaustively test exact combinations within those bounds.

## Key Results
* **Classification (Claim Occurrence):** 
  * F1 Score (Weighted): `[0.77]`
* **Regression (Claim Severity):** 
  * RMSE: `[RMSE: 8378.846093356431]`
  * MAE:  `[MAE : 3556.2627508324217]`

## Tech Stack
* **Algorithms:** XGBoost, Stochastic Gradient Descent (SGD)
* **Libraries:** Scikit-Learn, Pandas, NumPy, Statsmodels, Matplotlib, Seaborn
