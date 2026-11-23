# Credit Scoring Model

**Machine Learning Internship Project – CodeAlpha**

## 1. Project Overview

The Credit Scoring Model is a machine learning project designed to predict the likelihood of credit default based on customer financial information.

The project uses classification algorithms to analyse credit-related data, identify patterns, and predict whether a customer belongs to the default or non-default category.

## 2. Problem Statement

Financial institutions need reliable methods to assess the creditworthiness of customers. Traditional assessment methods can be time-consuming and may not efficiently identify potential credit risks.

This project aims to develop a machine learning model that analyses customer data and predicts credit default risk, supporting more informed credit assessment.

## 3. Technologies Used

- Programming Language: Python
- Platform: Google Colab
- Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Joblib
- Dataset: UCI Machine Learning Repository
- Machine Learning Approach: Classification

## 4. Machine Learning Algorithms

### Logistic Regression
- Used as a baseline classification algorithm.
- Predicts the probability of belonging to a particular class.

### Random Forest
- Uses multiple decision trees to make predictions.
- Combines predictions from individual trees.
- Achieved better overall performance than Logistic Regression in this experiment.

## 5. Model Performance and Evaluation

| Evaluation Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 67.97% | 79.60% |
| Precision | 36.72% | 53.95% |
| Recall | 62.02% | 53.05% |
| F1-score | 46.13% | 53.50% |
| ROC-AUC | 70.81% | 76.96% |

## 6. Results and Analysis

The Random Forest classifier achieved better overall results compared with Logistic Regression.

Key observations:

- Random Forest achieved 79.60% accuracy.
- Its F1-score was 53.50%, compared with 46.13% for Logistic Regression.
- Its ROC-AUC score was 76.96%.
- Logistic Regression achieved higher recall at 62.02%, compared with 53.05% for Random Forest.

Based on the F1-score, Random Forest was selected as the best-performing model.

## 7. Model Saving and Output

The trained model was saved using Joblib for future use.

The following project files were generated:

1. `credit_scoring_model.py` – Python implementation.
2. `credit_scoring_model.joblib` – Saved trained model.
3. `Credit_Scoring_Comparison.png` – Model performance comparison graph.

A sample prediction was also generated using a held-out test record.

## 8. Conclusion

This project demonstrates how machine learning classification algorithms can be applied to credit risk prediction.

By comparing Logistic Regression and Random Forest, the project identified Random Forest as the better-performing model based on F1-score.

The project provided practical experience in data preprocessing, model training, performance evaluation, model comparison, prediction, and model saving.

The developed model is an educational prototype and requires further validation, fairness testing, and real-world evaluation before any practical credit decision-making use.

## 9. Future Enhancements

- Improve model performance through hyperparameter tuning.
- Explore additional classification algorithms.
- Develop a user-friendly interface for credit risk prediction.
- Improve model interpretability and fairness evaluation.
- Deploy the model as a web-based application.

---

**Project:** Credit Scoring Model  
**Internship Organization:** CodeAlpha  
**Domain:** Machine Learning