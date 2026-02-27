from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from preprocess import preprocess
import os
import joblib

# Create models directory if it doesn't exist
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def main():
    X,y = preprocess()
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
    logReg = LogisticRegression(max_iter=1000,class_weight='balanced')
    # rfc_model = RandomForestClassifier(random_state=42)
    logReg.fit(X_train, y_train)
    # rfc_model.fit(X_train,y_train)

    y_pred = logReg.predict(X_test)
    # y_pred = rfc_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    confMat = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {acc:.3f}",)
    print(f"F1Score: {f1:.3f}",)
    print("Confusion Matrix:\n",confMat)

    # Save the model
    model_path = os.path.join(MODEL_DIR, "logistic_model.pkl")
    # model_path = os.path.join(MODEL_DIR, "randomForest_model.pkl")
    joblib.dump(logReg, model_path)
    # joblib.dump(rfc_model, model_path)

    print(f"Model saved at: {model_path}")

if __name__ == "__main__":
    main()



