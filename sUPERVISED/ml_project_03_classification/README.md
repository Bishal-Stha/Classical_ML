# Customer Churn Prediction Model

## Problem Statement

Create a Machine learning model to pedict the **Customer Churn ** From Kaggle.

### Dataset Source

The dataset is **Customer Churn** dataset and it is derived from [Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

### Model Selection

Logistic Regression is selected because:

- The dataset is medium.
- The problem is binary classification.
- It is interpretable.
- Dataset is not balanced.
- It serves as a strong baseline model.
Other models like Random Forest, SVM, XGBoost, etc may be tested later for performance comparison.

### Metrics I will use

- Accuracy
- F1 Score
- Confusion matrix

### Critical Preprocessing

1. Numerical Columns

- tenure
- MonthlyCharges
- TotalCharges

2. Categorical Columns

- customerID
- gender
- SeniorCitizen
- Partner
- Dependents
- PhoneService
- MultipleLines
- InternetService
- OnlineSecurity
- OnlineBackup
- DeviceProtection
- TechSupport
- StreamingTV
- StreamingMovies
- Contract
- PaperlessBilling
- PaymentMethod
- Churn

3. Clearly useless columns

- customerID
- gender (No of males and females is nearly equal so it won't affect significantly.)
- 
- 

4. Columns that need encoding

- SeniorCitizen
- gender (Yes, No) ✅
- Partner (Yes, No) ✅
- Dependents (Yes, No) ✅
- PhoneService  (Yes, No) ✅
- MultipleLines (No, Yes, No phone service)☑️
- InternetService (Fiber optic, DSL, No)☑️
- OnlineSecurity (No, Yes, No internet service) ✅
- OnlineBackup (No, Yes, No internet service) ✅
- DeviceProtection (No, Yes, No internet service) ✅
- TechSupport (No, Yes, No internet service) ✅
- StreamingTV (No, Yes, No internet service) ✅
- StreamingMovies (No, Yes, No internet service) ✅
- Contract (Month-to-month, Two year, One year)☑️
- PaperlessBilling (Yes, No) ✅
- PaymentMethod (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))☑️
- Churn (Yes, No) ✅

5. Missing Value Handling

- None. There are no missing or duplicated values.

## Project Structure

project/
├── data/
├── models/
│   └── Iris_model.pkl
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   └── train.py
├── README.md
├── requirments.txt
