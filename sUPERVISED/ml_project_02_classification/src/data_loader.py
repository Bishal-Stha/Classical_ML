import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(BASE_DIR)
FILE_DIR = os.path.join(BASE_DIR,"data","Iris.csv")
# print(FILE_DIR)

def load_data(path):
    return pd.read_csv(path)

def main():
    df = load_data(FILE_DIR)
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {df.columns}")
    print(f"\nContents of Dataset\n{df.head()}")
    print(f"\nNull check.\n{df.isnull().sum()}")
    print(f"\nDuplicate check.\n{df.duplicated().sum()}")
    print(f"\nDtypes\n{df.dtypes}")
    print(f"\nNumerical data summary: \n {df.describe()}")
    print(f"\nCategorical data summary: \n {df.describe(include='object')}") # type: ignore
    print(f"Values: {df['Species'].unique()}")

if __name__ == "__main__":
    main()

