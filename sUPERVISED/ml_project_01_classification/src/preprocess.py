# from data_loader import BASE_DIR, load_data
# import os
# import pandas as pd

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# DATA_PATH = os.path.join(BASE_DIR, "data", "train.csv")

# # load the data.
# df = load_data(DATA_PATH)

# # drop useless columns 
# df = df.drop(columns=["PassengerId","Name","Cabin","Ticket"])
# print(df.shape)
# print(df.columns)

# print("\nBefore Filling the Null values.\n",df.head(10))

# # Fill Age with median and Emarked with mode.
# df["Age"] = df["Age"].fillna(df["Age"].median())
# df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode())
# print("\nAfter Filling the Null values.\n",df.head(10))

# # Perform Encoding on Sex and Embarked.
# df["Sex"] = df["Sex"].map({"female": 0, "male": 1})

# # One-hot encode Embarked
# df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

# print("\nAfter Encoding:\n", df.head())



# X = df.drop("Survived", axis=1)
# y = df["Survived"]

# print("\nFeature shape:", X.shape)
# print("Target shape:", y.shape)

### Converted everything into a function.
import os
import pandas as pd
from data_loader import load_data

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "train.csv")

def preprocess():
    df = load_data(DATA_PATH)

    df = df.drop(columns=["PassengerId", "Name", "Cabin", "Ticket"])

    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    df["Sex"] = df["Sex"].map({"female": 0, "male": 1})

    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    return X, y