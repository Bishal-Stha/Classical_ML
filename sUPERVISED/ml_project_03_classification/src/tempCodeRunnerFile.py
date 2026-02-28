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

    print