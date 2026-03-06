from data_loader import load_data, visualize
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR,"data","Wholesale customers data.csv")

df = load_data(DATA_PATH)

# Displaying head data
print(df.head())

print(f"\nShape: {df.shape}")
print(f"\nName of columns:\n{df.columns}")
print(f"\nNull values:\n{df.isna().sum()}")
print(f"\nNumerical features:\n{df.describe()}")
# print(f"\nCategorical features:\n{df.describe(include='object')}") # type: ignore
# dataset doesn't have any categorical features.

scaler = StandardScaler()
pca = PCA(n_components=2)

def preprocess():
    # print(X_pca_scaled.shape)
    X_scaled = scaler.fit_transform(df)
    X_pca_scaled = pca.fit_transform(X_scaled)
    return X_pca_scaled