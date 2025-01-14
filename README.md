# BO vs CMA-ES for Hyperparameter Optimization in XGBoost

## 1. Introduction
This project compares Bayesian Optimization (BO) using Optuna with Covariance Matrix Adaptation Evolution Strategy (CMA-ES) for hyperparameter optimization in XGBoost models. The objective was to evaluate the performance of both optimization techniques across multiple datasets and assess their effectiveness, consistency, and convergence speed.

---

## 2. Literature Review and Problem Formulation
1. **Objective**: We decided to focus on **classification tasks** using the following metrics:
   - Accuracy
   - F1 Score
   - Recall
   - Precision
2. **Dataset**: For initial experiments, we selected the [Mushroom Classification Dataset](https://www.kaggle.com/datasets/uciml/mushroom-classification) from Kaggle.
3. **Optimization Tools**: We chose the following libraries:
   - **Optuna** (for Bayesian Optimization)
   - **cma** (for CMA-ES)

---

## 3. Algorithm Implementation and Initial Experiments
1. **Model Selection**: We used a **LightGBM** model for training on the Mushroom dataset.
2. **Hyperparameter Space**:
   - `num_leaves`
   - `min_data_in_leaf`
   - `learning_rate`
   - `feature_fraction`
   - `bagging_fraction`
   - `bagging_freq`
   - `lambda_l1`
   - `lambda_l2`
3. **Initial Benchmark**:
   - Randomly chosen parameters yielded **99.75% accuracy** on the Mushroom dataset (due to its small size and simplicity).
4. **Optimization Results**:
   - **Optuna** achieved **100% accuracy**.
   - **CMA-ES** also achieved **100% accuracy**.
5. **Feature Importance**:
   - The most critical hyperparameters were:
     - `feature_fraction`
     - `lambda_l1`
     - `lambda_l2`
     - `min_data_in_leaf`
     - `learning_rate`

---

## 4. Evaluation Pipeline and Comparison
1. **Pipeline Improvements**:
   - Modified the codebase to support **any classification dataset**.
2. **Datasets Used**:
   - Mushroom Classification
   - Heart Attack Prediction
   - Mobile Device Usage and User Behavior
   - Mobile Price Classification
   - Loan Approval Classification
   - Beaches vs Mountains Preference
3. **Comparison Results**:
   - Both Optuna and CMA-ES outperformed the benchmark model across all datasets.
   - **CMA-ES** consistently delivered slightly better results than Optuna.
4. **Consistency Tests**:
   - We tested consistency by running models with **five different random seeds**.
   - Results:
     - **Optuna** showed greater sensitivity to seed changes.
     - **CMA-ES** exhibited only minor variations.
5. **Performance Metrics**:
   - Optimized models (both Optuna and CMA-ES) consistently surpassed the benchmark in terms of accuracy, F1 score, and recall.

---

## 5. Results
1. **Parameter Importance**:
   - For Optuna:
     - `learning_rate` was typically the most critical parameter.
     - Other important parameters included `feature_fraction`, `lambda_l2`, and `bagging_freq`.
2. **Convergence**:
   - **Optuna** tended to converge faster than CMA-ES.
3. **Performance**:
   - On average, **CMA-ES** achieved higher accuracy than Optuna.

---

## 6. Conclusion
- Both optimization techniques proved highly effective for hyperparameter tuning in XGBoost models.
- **CMA-ES** emerged as the more consistent performer, achieving slightly better results across datasets and showing less sensitivity to seed variations.
- **Optuna** demonstrated faster convergence, making it a suitable choice when time is a critical factor.
