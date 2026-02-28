from preprocess import preprocess
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import joblib
import os

# Create models directory if it doesn't exist
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def train():
    X,y = preprocess()
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size=0.3)
    kNN = KNeighborsClassifier(n_neighbors=5, )
    kNN.fit(X_train, y_train)

    y_pred = kNN.predict(X_test)

    ### Test metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    confMat = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {acc:.3f}")
    print(f"F1Score: {f1:.3f}")
    print(f"Confusion Matrix\n{confMat}")

    # Save the model
    model_path = os.path.join(MODEL_DIR, "Iris_model.pkl")
    joblib.dump(kNN, model_path)

    print(f"Model saved at: {model_path}")

if __name__ == "__main__":
    train()