# Customer Churn Prediction

A machine learning project designed to predict whether a telecom customer is likely to churn.

This project uses the IBM Telco Customer Churn dataset and follows a full ML workflow: data cleaning, exploratory data analysis, preprocessing, model comparison, threshold tuning, explainability, and deployment via a Streamlit app.

## Overview

The app accepts customer attributes such as:

- Tenure
- Contract type
- Internet service
- Monthly charges
- Total charges
- Payment method
- Online security
- Tech support
- Streaming services

It then provides:

- Churn probability
- Churn / No Churn prediction
- SHAP-based explanation of the result
- Feature contribution breakdown

---

## Dataset

The project uses the IBM Telco Customer Churn dataset.

The dataset contains 7,043 customers and 21 columns, including customer information, subscribed services, account-level details, and the final `Churn` label.

### Target variable

```text
Churn
No  -> Customer stayed
Yes -> Customer left
```

### Class distribution

The dataset is slightly imbalanced:

- No: 73.46%
- Yes: 26.54%

### Key EDA findings

A few patterns stood out during analysis:

- Month-to-month customers had much higher churn than one-year or two-year contract customers.
- Customers with shorter tenure were more likely to churn.
- Electronic check users showed a relatively high churn rate.
- Fiber optic customers had higher churn than DSL customers in this dataset.
- Customers with higher monthly charges showed higher average churn rates.

> These are observations from the dataset and should not be interpreted as proof that these features are direct causes of churn.

---

## Data preprocessing

Before training the models, the project:

- Removed `customerID` because it is an identifier rather than a useful predictive feature.
- Converted `TotalCharges` from object to numeric.
- Handled missing blank values in `TotalCharges`.
- Applied one-hot encoding to categorical variables.
- Scaled numerical features.
- Kept the preprocessing inside a scikit-learn pipeline so the same transformations are used during prediction.

This produced 45 model features in the final transformed dataset.

---

## Models evaluated

I compared three models:

- Logistic Regression
- Random Forest
- XGBoost

Cross-validation was used to compare performance on the training data.

| Model | Mean CV F1 |
| --- | ---: |
| Logistic Regression | 0.595 |
| XGBoost | 0.583 |
| Random Forest | 0.560 |

Based on the CV results, I selected Logistic Regression as the final model.

---

## Threshold tuning

The default classification threshold of 0.50 was not ideal for this problem because missing a likely churner may matter more than maximizing overall accuracy.

I generated out-of-fold predictions on the training data and selected a threshold of:

```text
0.30
```

The final threshold was then evaluated once on the held-out test set.

### Final test results at the 0.30 threshold

| Metric | Value |
| --- | ---: |
| Accuracy | 0.743 |
| Precision | 0.511 |
| Recall | 0.759 |
| F1 Score | 0.611 |
| ROC-AUC | 0.836 |

The lower threshold improves recall for the churn class and helps catch more customers who actually churn.

---

## Model explainability

SHAP was used to understand why a model made a specific prediction.

The Streamlit app highlights the main features contributing toward or away from churn for each customer.

Some of the important features identified during analysis include:

- Tenure
- Total Charges
- Monthly Charges
- Contract
- Internet Service
- Payment Method
- Streaming Services

For example, a customer with very low tenure and a month-to-month contract may receive a strong contribution toward churn.

> SHAP explanations help interpret model behavior but should not be treated as proof that a feature causes churn.

---

## Streamlit app

The trained model is saved as:

```text
customer_churn_model.pkl
```

This saved artifact contains:

- Trained preprocessing + model pipeline
- Decision threshold
- SHAP background data

The app loads this object and uses the same pipeline for new customer predictions.

---

## Project structure

```text
customer-churn-prediction/
├── app.py
├── customer_churn_model.pkl
├── requirements.txt
├── .gitignore
├── README.md
└── screenshots/
```

---

## Run locally

1. Clone the repository:

```bash
git clone <your-repository-url>
cd customer-churn-prediction
```

2. Create a virtual environment:

```bash
python -m venv venv
```

3. Activate it on Windows:

```bash
venv\Scripts\activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Run the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your browser.

---

## Tech stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Streamlit
- Joblib

---

## What I learned

This was my first complete ML project where I went beyond training a model in a notebook and built it into an end-to-end workflow.

The main areas I worked on included:

- Exploratory data analysis
- Data cleaning
- Categorical encoding
- Feature scaling
- ML pipelines
- Model comparison
- Cross-validation
- Classification threshold tuning
- Model evaluation
- SHAP explainability
- Saving and loading ML models
- Building a simple Streamlit app

---

## Future improvements

Some ideas for later improvements:

- Better probability calibration
- More detailed model monitoring
- Improved UI/UX
- Deployment to a production environment
- More extensive hyperparameter tuning
- Testing the model on newer customer data
