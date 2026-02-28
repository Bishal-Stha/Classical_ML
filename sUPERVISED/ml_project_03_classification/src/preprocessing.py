import os
import pandas as pd
from data_loader import load_data

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR,"data","customer_churn.csv")

def preprocessing():
    df = load_data(DATA_PATH)
    df = df.drop(columns=["customerID"]) # dropped useless feature.
    internet_cols = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies"
    ]

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()  # Drop rows with missing values
    all_yesNo_col = ["gender","Partner","Dependents","PhoneService","PaperlessBilling","Churn",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies","MultipleLines"]

    df["MultipleLines"] = df["MultipleLines"].replace({"No phone service": "No"})

    df[internet_cols] = df[internet_cols].replace(
        {"No internet service": "No"}
    )

    df[all_yesNo_col] = df[all_yesNo_col].replace(
        {"Yes": 1, "No": 0, "Male":1, "Female":0}
    )

    df["Contract"] = df["Contract"].replace({"Month-to-month":0, "Two year": 1, "One year": 2})
    df = pd.get_dummies(df, columns=["InternetService", "PaymentMethod"], drop_first=True)
    # print(df.head())
    X,y = df.drop(columns=["Churn"]), df["Churn"]
    return X,y

if __name__ == "__main__":
    preprocessing()