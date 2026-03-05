from data_loader import load_data, visualize
# import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from kneed import KneeLocator

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
X_scaled = scaler.fit_transform(df)
X_scaled_pca = pca.fit_transform(X_scaled)
print(X_scaled_pca.shape)

wcss = []
for k in range(1,11):
    kmeans = KMeans(n_clusters=k, init='k-means++')
    kmeans.fit(X_scaled_pca)
    wcss.append(kmeans.inertia_)

kl = KneeLocator(range(2,11), wcss, curve='convex', direction='decreasing', )

kmeans = KMeans(n_clusters=kl, init='k-means++')
y_labels = kmeans.fit_predict
y_test_labels = kmeans.predict()
# visualize(X_scaled_pca[:,0],X_scaled_pca[:,1])





