
---

# Telco Customer Churn Prediction Dashboard

## a. Problem Statement

Customer churn occurs when customers stop doing business with a company. In the telecommunications industry, retaining existing customers is often more cost-effective than acquiring new ones. The goal of this project is to build a machine learning system that predicts the likelihood of a customer leaving based on their demographics, account information, and service usage. By identifying "at-risk" customers, the company can take proactive retention measures.

## b. Dataset Description

The project uses the **IBM Telco Customer Churn** dataset.

* **Target Variable:** `Churn` (Yes/No - converted to 1/0).
* **Instance Size:** 7,043 records.
* **Feature Size:** 20 features (expanded to 30+ after encoding).
* **Key Features:** * **Demographics:** Gender, Senior Citizen status, Partners, Dependents.
* **Services:** Phone, Multiple lines, Internet (Fiber optic/DSL), Online Security, Streaming TV.
* **Account Info:** Tenure, Contract type (Month-to-month, One year, Two year), Payment method, Paperless billing, Monthly Charges, and Total Charges.

---

## c. Models Used & Comparison

I implemented six different classification models to compare their effectiveness in predicting churn.

### Evaluation Metrics Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| **Logistic Regression** | 0.8211 | 0.8622 | 0.6862 | 0.5979 | 0.6390 | 0.5230 |
| **Decision Tree** | 0.8062 | 0.8482 | 0.7049 | 0.4611 | 0.5575 | 0.4566 |
| **kNN** | 0.7722 | 0.7971 | 0.5765 | 0.5255 | 0.5498 | 0.3985 |
| **Naive Bayes** | 0.6657 | 0.8377 | 0.4359 | 0.8928 | 0.5858 | 0.4222 |
| **Random Forest (Ensemble)** | 0.7913 | 0.8368 | 0.6479 | 0.4638 | 0.5406 | 0.4200 |
| **XGBoost (Ensemble)** | 0.7913 | 0.8408 | 0.6270 | 0.5228 | 0.5702 | 0.4370 |

---

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| --- | --- |
| **Logistic Regression** | **Top Performer:** Achieved the highest Accuracy and MCC. It handled the linear relationships in the data very effectively. |
| **Decision Tree** | **Overfitting Risk:** While it had good precision, its recall was lower, suggesting it struggles to generalize for all churn cases. |
| **kNN** | **Distance Sensitivity:** Even with scaling, kNN underperformed, likely due to the "curse of dimensionality" after one-hot encoding. |
| **Naive Bayes** | **Recall Specialist:** It identified nearly 90% of churners (highest Recall) but at the cost of many false positives (lowest Precision). |
| **Random Forest** | **Stable but Average:** Provided consistent results but didn't outperform the simpler Logistic Regression on this specific dataset. |
| **XGBoost** | **Strong Ensemble:** Showed better balance between metrics than Random Forest, proving effective at capturing non-linear patterns. |

---
