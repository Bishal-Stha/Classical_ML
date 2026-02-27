import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "train.csv")

def load_data(path):
    return pd.read_csv(path)

def main():
    train = load_data(DATA_PATH)

    print("Shape:", train.shape)
    print("\nColumn Names:\n", train.columns)
    print("\nFirst 5 Rows:\n", train.head())
    print("\nMissing Values Per Column:\n", train.isna().sum())
    print("\nData Types:\n", train.dtypes)

    print("\nStatistical Summary (Numerical Columns):\n", train.describe())

    print("\nStatistical Summary (Categorical Columns):\n", train.describe(include="object")) # type: ignore

if __name__ == "__main__":
    main()