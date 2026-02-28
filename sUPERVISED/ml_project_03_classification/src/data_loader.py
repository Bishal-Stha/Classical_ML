import pandas as pd
import os

def load_data(path):
    return pd.read_csv(path)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR,"data","customer_churn.csv")
# print(DATA_PATH)

# ##### Data Lookup.
def main():
    df = load_data(DATA_PATH)
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {df.columns}")
    print(f"\nNumerical columns: {df.describe()}")
    print(f"\nCategorical columns: {df.describe(include='object')}") # type: ignore
    print(f"\nData lookup:\n{df.sample(5)}")

if __name__ == "__main__":
    main()