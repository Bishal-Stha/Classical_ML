import streamlit as st
import joblib
import pandas as pd
import os

# Load model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "logistic_model.pkl")
model = joblib.load(model_path)

st.title("Titanic Survival Prediction",text_alignment="center")

# -------------------------------
# User Inputs
# -------------------------------
Pclass = st.selectbox("Passenger Class", [1, 2, 3])
Sex = st.selectbox("Sex", ["male", "female"])
Age = st.slider("Age", 0, 80, 25)
SibSp = st.number_input("Siblings/Spouses", 0, 10, 0)
Parch = st.number_input("Parents/Children", 0, 10, 0)
Fare = st.number_input("Fare", 0.0, 500.0, 50.0)

# -------------------------------
# Preprocess input
# -------------------------------
# Encode Sex
Sex = 1 if Sex == "male" else 0

# Embarked columns: default to 0 (assume 'C' if user input is removed)
input_df = pd.DataFrame({
    "Pclass": [Pclass],
    "Sex": [Sex],
    "Age": [Age],
    "SibSp": [SibSp],
    "Parch": [Parch],
    "Fare": [Fare],
    "Embarked_Q": [0],
    "Embarked_S": [0]
})

# -------------------------------
# Predict
# -------------------------------
if st.button("Predict"):
    prediction = model.predict(input_df)
    result = "Survived" if prediction[0] == 1 else "Did Not Survive"
    st.success(f"Prediction: {result}")