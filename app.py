import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Heart Failure Prediction",
    page_icon="❤️",
    layout="centered"
)

# -----------------------------
# Load Model & Transformer
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    poly = joblib.load("polynomial_features_transformer.pkl")
    return model, poly

model, poly = load_artifacts()

# -----------------------------
# Title & Description
# -----------------------------
st.title("❤️ Heart Failure Prediction System")

st.markdown("""
This application predicts the **risk of heart failure** using a  
**Gradient Boosting Machine Learning model** optimized for **high recall**.

⚠️ *For educational purposes only. Not a medical diagnosis.*
""")

st.divider()

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("🧑‍⚕️ Patient Information")

age = st.sidebar.slider(
    "Age (years)", 30, 95, 60,
    help="Patient's age in years"
)

sex = st.sidebar.radio(
    "Sex",
    ["Male", "Female"],
    help="Biological sex of the patient"
)
sex = 1 if sex == "Male" else 0

anaemia = st.sidebar.radio(
    "Anaemia",
    ["No", "Yes"],
    help="Decrease of red blood cells or hemoglobin"
)
anaemia = 1 if anaemia == "Yes" else 0

diabetes = st.sidebar.radio(
    "Diabetes",
    ["No", "Yes"],
    help="Whether the patient has diabetes"
)
diabetes = 1 if diabetes == "Yes" else 0

high_blood_pressure = st.sidebar.radio(
    "High Blood Pressure",
    ["No", "Yes"],
    help="If the patient has hypertension"
)
high_blood_pressure = 1 if high_blood_pressure == "Yes" else 0

smoking = st.sidebar.radio(
    "Smoking",
    ["No", "Yes"],
    help="Whether the patient smokes"
)
smoking = 1 if smoking == "Yes" else 0

creatinine_phosphokinase = st.sidebar.number_input(
    "Creatinine Phosphokinase (mcg/L)",
    min_value=20,
    max_value=8000,
    value=250,
    help="Enzyme released when muscle tissue is damaged"
)

ejection_fraction = st.sidebar.slider(
    "Ejection Fraction (%)",
    10, 80, 38,
    help="Percentage of blood leaving the heart at each contraction"
)

platelets = st.sidebar.number_input(
    "Platelets (kiloplatelets/mL)",
    min_value=100000,
    max_value=900000,
    value=250000,
    help="Platelet count in blood"
)

serum_creatinine = st.sidebar.number_input(
    "Serum Creatinine (mg/dL)",
    min_value=0.5,
    max_value=10.0,
    value=1.2,
    help="Level of creatinine in blood (kidney function indicator)"
)

serum_sodium = st.sidebar.slider(
    "Serum Sodium (mEq/L)",
    110, 150, 137,
    help="Sodium level in blood"
)

time = st.sidebar.slider(
    "Follow-up Period (days)",
    1, 300, 130,
    help="Duration of patient follow-up"
)

# -----------------------------
# Input Summary
# -----------------------------
input_data = pd.DataFrame([{
    "age": age,
    "anaemia": anaemia,
    "creatinine_phosphokinase": creatinine_phosphokinase,
    "diabetes": diabetes,
    "ejection_fraction": ejection_fraction,
    "high_blood_pressure": high_blood_pressure,
    "platelets": platelets,
    "serum_creatinine": serum_creatinine,
    "serum_sodium": serum_sodium,
    "sex": sex,
    "smoking": smoking,
    "time": time
}])

st.subheader("📋 Patient Data Summary")
st.dataframe(input_data, use_container_width=True)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Heart Failure Risk"):
    input_poly = poly.transform(input_data)
    probability = model.predict_proba(input_poly)[0][1]

    threshold = 0.25
    prediction = 1 if probability >= threshold else 0

    st.divider()
    st.subheader("📊 Prediction Result")

    st.metric(
        label="Predicted Risk Probability",
        value=f"{probability * 100:.2f}%"
    )

    if prediction == 1:
        st.error(
            "⚠️ **High Risk of Heart Failure Detected**\n\n"
            "The model indicates a **higher likelihood** of heart failure. "
            "Please consult a qualified healthcare professional."
        )
    else:
        st.success(
            "✅ **Low Risk of Heart Failure Detected**\n\n"
            "The model indicates a **lower likelihood** of heart failure."
        )

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption(
    "Developed by **Albert Antony S** | "
    "Heart Failure Prediction – Machine Learning Project"
)
