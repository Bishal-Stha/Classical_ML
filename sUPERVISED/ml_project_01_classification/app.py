import streamlit as st
import joblib
import pandas as pd
import os
from datetime import datetime

# -------------------------------
# Paths and model load
# -------------------------------
BASE_DIR = os.path.abspath(".")
model_path = os.path.join(BASE_DIR, "models", "logistic_model.pkl")
logs_dir = os.path.join(BASE_DIR, "logs")
os.makedirs(logs_dir, exist_ok=True)
prediction_log_file = os.path.join(logs_dir, "prediction_log.csv")
error_log_file = os.path.join(logs_dir, "error_log.csv")

# Load model safely
try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("Titanic Survival Prediction", text_alignment="center")

# -------------------------------
# User Inputs
# -------------------------------
st.subheader("Passenger Information")

Pclass = st.selectbox("Passenger Class (1=1st, 2=2nd, 3=3rd)", [1, 2, 3])
Sex = st.selectbox("Sex", ["male", "female"])
Age = st.slider("Age (years)", 0, 120, 25)
SibSp = st.number_input("Number of Siblings/Spouses aboard", 0, 10, 0)
Parch = st.number_input("Number of Parents/Children aboard", 0, 10, 0)
Fare = st.number_input("Passenger Fare ($)", 0.0, 600.0, 50.0, step=0.1)
Embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# -------------------------------
# Preprocess input
# -------------------------------
Sex_encoded = 1 if Sex == "male" else 0
Embarked_Q = 1 if Embarked == "Q" else 0
Embarked_S = 1 if Embarked == "S" else 0

input_df = pd.DataFrame({
    "Pclass": [Pclass],
    "Sex": [Sex_encoded],
    "Age": [Age],
    "SibSp": [SibSp],
    "Parch": [Parch],
    "Fare": [Fare],
    "Embarked_Q": [Embarked_Q],
    "Embarked_S": [Embarked_S]
})

# -------------------------------
# Logging functions
# -------------------------------
def log_prediction(input_df, prediction, probability):
    df_copy = input_df.copy()
    df_copy["Prediction"] = prediction[0]
    df_copy["Probability"] = probability
    df_copy["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df_copy.to_csv(prediction_log_file, mode='a', index=False, header=not os.path.isfile(prediction_log_file))

def log_error(error_msg):
    with open(error_log_file, "a") as f:
        f.write(f"{datetime.now()} - {error_msg}\n")

# -------------------------------
# Predict
# -------------------------------
if st.button("Predict"):
    try:
        # Input validation
        if Age < 0 or Age > 120:
            st.warning("Age should be between 0 and 120.")
        elif SibSp < 0 or SibSp > 10:
            st.warning("Siblings/Spouses should be between 0 and 10.")
        elif Parch < 0 or Parch > 10:
            st.warning("Parents/Children should be between 0 and 10.")
        elif Fare < 0 or Fare > 600:
            st.warning("Fare should be between $0 and $600.")
        else:
            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df)[0][1]
            result = "Survived" if prediction[0] == 1 else "Did Not Survive"

            st.success(f"Prediction: {result}")
            st.info(f"Predicted probability of survival: {probability:.2f}")

            # Log the prediction
            log_prediction(input_df, prediction, probability)

    except Exception as e:
        error_msg = f"Prediction error: {e}"
        st.error("An error occurred during prediction. Check logs for details.")
        log_error(error_msg)