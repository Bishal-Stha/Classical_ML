import os
from data_loader import load_data

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR,"data","Iris.csv")

def preprocess():
    df = load_data(DATA_PATH)
    df = df.drop(columns=["Id"])
    df["Species"] = df["Species"].map({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}) #Values: ['Iris-setosa' 'Iris-versicolor' 'Iris-virginica']
    # df = pd.get_dummies(df, columns=["Species"], drop_first=True)
    # print(df.sample(5))
    # print(df.shape)
    # print(df.columns)
    X,y = df.drop(columns=["Species"]), df["Species"]
    return X,y

# if __name__ == "__main__":
#     preprocess()