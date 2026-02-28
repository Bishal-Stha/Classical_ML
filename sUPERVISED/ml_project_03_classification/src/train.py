from preprocessing import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import joblib
import os

# Create models directory if it doesn't exist
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def train():
    X, y = preprocessing()
    
    # Check for missing values
    if X.isnull().any().any() or y.isnull().any():
        print("Warning: Missing values detected. Dropping rows with NaN values.")
        valid_idx = X.notna().all(axis=1) & y.notna()
        X = X[valid_idx]
        y = y[valid_idx]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # print(X_train.dtypes)
    churnModel = LogisticRegression(random_state=42, class_weight="balanced")

    churnModel.fit(X_train, y_train)
    y_pred = churnModel.predict(X_test)

    acc = accuracy_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    confMat = confusion_matrix(y_test,y_pred)

    print(f"Accuracy: {acc}")
    print(f"F1 Score: {f1}")
    print(f"Confusion Matrix:\n {confMat}")

    # Save the model
    model_path = os.path.join(MODEL_DIR, "Customer_Churn_Prediction_Model.pkl")
    joblib.dump(churnModel, model_path)

    print(f"Model saved at: {model_path}") 

if __name__ == "__main__":
    train()