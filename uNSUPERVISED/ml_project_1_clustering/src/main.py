import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from preprocess import preprocess

def main():
    X_pca_scaled = preprocess()
    dbscan = DBSCAN(eps=0.5, metric='euclidean', min_samples=5)
    dbscan.fit(X_pca_scaled)
    y_pred = dbscan.fit_predict(X_pca_scaled)
    # print(dbscan.labels_.shape, X_scaled.shape)

    score = silhouette_score(X_pca_scaled, y_pred)
    print(f"Silhouette Score: {score:.3f}")


    plt.scatter(X_pca_scaled[:,0], X_pca_scaled[:,1],c=dbscan.labels_)
    plt.title("Wholesale customers data")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    # plt.savefig("Wholesale customers data.jpg")
    plt.show()

if __name__ == "__main__":
    main()

