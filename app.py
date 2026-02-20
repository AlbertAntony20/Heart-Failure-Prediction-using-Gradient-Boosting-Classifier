import streamlit as st
import numpy as np
import joblib

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Heart Failure Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Failure Prediction System")
st.write(
    "This application predicts the risk of heart failure using a trained "
    "Gradient Boosting model. The system is optimized to prioritize recall, "
    "which is crucial for healthcare-related predictions."
)

# --------------------------------------------------
# Load Model and Polynomial Transformer
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    poly = joblib.load("polynomial_features_transformer.pkl")
    return model, poly

model, poly = load_artifacts()

# --------------------------------------------------
# User Input Section
# --------------------------------------------------
st.subheader("🧑‍⚕️ Patient Information")

age = st.slider("Age", min_value=20, max_value=100, value=50)

anaemia = st.selectbox("Anaemia", ["No", "Yes"])

creatinine_phosphokinase = st.number_input(
    "Creatinine Phosphokinase (mcg/L)",
    min_value=10,
    max_value=8000,
    value=250
)

diabetes = st.selectbox("Diabetes", ["No", "Yes"])

ejection_fraction = st.slider(
    "Ejection Fraction (%)",
    min_value=10,
    max_value=80,
    value=40
)

high_blood_pressure = st.selectbox("High Blood Pressure", ["No", "Yes"])

platelets = st.number_input(
    "Platelets (kiloplatelets/mL)",
    min_value=100000,
    max_value=900000,
    value=250000
)

serum_creatinine = st.number_input(
    "Serum Creatinine (mg/dL)",
    min_value=0.5,
    max_value=10.0,
    value=1.2,
    step=0.1
)

serum_sodium = st.slider(
    "Serum Sodium (mEq/L)",
    min_value=110,
    max_value=150,
    value=137
)

sex = st.selectbox("Sex", ["Female", "Male"])

smoking = st.selectbox("Smoking", ["No", "Yes"])

time = st.number_input(
    "Follow-up Period (days)",
    min_value=1,
    max_value=300,
    value=120
)

# --------------------------------------------------
# Prepare Input for Model (IMPORTANT: ORDER MATTERS)
# --------------------------------------------------
input_array = np.array([[
    age,
    1 if anaemia == "Yes" else 0,
    creatinine_phosphokinase,
    1 if diabetes == "Yes" else 0,
    ejection_fraction,
    1 if high_blood_pressure == "Yes" else 0,
    platelets,
    serum_creatinine,
    serum_sodium,
    1 if sex == "Male" else 0,
    1 if smoking == "Yes" else 0,
    time
]])

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🔍 Predict Heart Failure Risk"):
    transformed_input = poly.transform(input_array)
    probability = model.predict_proba(transformed_input)[0][1]

    THRESHOLD = 0.25  # chosen to prioritize recall
    prediction = probability >= THRESHOLD

    st.subheader("📊 Prediction Result")

    if prediction:
        st.error(
            f"⚠️ **High Risk of Heart Failure Detected**\n\n"
            f"**Predicted Probability:** {probability:.2f}"
        )
    else:
        st.success(
            f"✅ **Low Risk of Heart Failure**\n\n"
            f"**Predicted Probability:** {probability:.2f}"
        )

    st.info(
        "📌 *Note:* A lower threshold (0.25) is used to reduce false negatives, "
        "which is critical in healthcare applications."
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("Developed by Albert Antony | B.Tech AI & Data Science")
