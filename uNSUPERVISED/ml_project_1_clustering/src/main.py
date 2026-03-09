import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from preprocess import preprocess
import os
import joblib

# Create models directory if it doesn't exist
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    X_pca_scaled = preprocess()
    dbscan = DBSCAN(eps=0.5, metric='euclidean', min_samples=5)
    dbscan.fit(X_pca_scaled)
    y_pred = dbscan.fit_predict(X_pca_scaled)
    # print(dbscan.labels_.shape, X_scaled.shape)

    score = silhouette_score(X_pca_scaled, y_pred)
    print(f"Silhouette Score: {score:.3f}")

    # Save the model
    model_path = os.path.join(MODEL_DIR, "Customer_segmentation.pkl")
    # model_path = os.path.join(MODEL_DIR, "randomForest_model.pkl")
    joblib.dump(dbscan, model_path)
    # joblib.dump(rfc_model, model_path)


    plt.scatter(X_pca_scaled[:,0], X_pca_scaled[:,1],c=dbscan.labels_)
    plt.title("Wholesale customers data")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    # plt.savefig("Wholesale customers data.jpg")
    plt.show()

if __name__ == "__main__":
    main()

