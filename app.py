import streamlit as st
import numpy as np
import pandas as pd
import joblib

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="Heart Failure Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Failure Prediction System")
st.write("Predict the risk of heart failure using a trained Gradient Boosting model.")

# -------------------- Load Model & Transformer --------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    poly = joblib.load("polynomial_features_transformer.pkl")
    return model, poly

model, poly = load_artifacts()

# -------------------- Input UI --------------------
st.subheader("🧑‍⚕️ Patient Details")

age = st.slider("Age", 20, 100, 50)
anaemia = st.selectbox("Anaemia", ["No", "Yes"])
creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase (mcg/L)", 10, 8000, 250)
diabetes = st.selectbox("Diabetes", ["No", "Yes"])
ejection_fraction = st.slider("Ejection Fraction (%)", 10, 80, 40)
high_blood_pressure = st.selectbox("High Blood Pressure", ["No", "Yes"])
platelets = st.number_input("Platelets (kiloplatelets/mL)", 100000, 900000, 250000)
serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", 0.5, 10.0, 1.2)
serum_sodium = st.slider("Serum Sodium (mEq/L)", 110, 150, 137)
sex = st.selectbox("Sex", ["Female", "Male"])
smoking = st.selectbox("Smoking", ["No", "Yes"])
time = st.number_input("Follow-up Period (days)", 1, 300, 120)

# -------------------- Encode Inputs --------------------
input_data = pd.DataFrame([{
    "age": age,
    "anaemia": 1 if anaemia == "Yes" else 0,
    "creatinine_phosphokinase": creatinine_phosphokinase,
    "diabetes": 1 if diabetes == "Yes" else 0,
    "ejection_fraction": ejection_fraction,
    "high_blood_pressure": 1 if high_blood_pressure == "Yes" else 0,
    "platelets": platelets,
    "serum_creatinine": serum_creatinine,
    "serum_sodium": serum_sodium,
    "sex": 1 if sex == "Male" else 0,
    "smoking": 1 if smoking == "Yes" else 0,
    "time": time
}])

# -------------------- Prediction --------------------
if st.button("🔍 Predict Risk"):
    transformed_input = poly.transform(input_data)
    probability = model.predict_proba(transformed_input)[0][1]

    THRESHOLD = 0.25
    prediction = 1 if probability >= THRESHOLD else 0

    st.subheader("📊 Result")

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Failure\n\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ Low Risk of Heart Failure\n\nProbability: {probability:.2f}")

    st.caption("Threshold set to 0.25 to prioritize recall (minimize false negatives).")

# -------------------- Footer --------------------
st.markdown("---")
st.caption("Developed by Albert Antony | AI & DS")
