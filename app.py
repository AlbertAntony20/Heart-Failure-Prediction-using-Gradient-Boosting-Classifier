import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Load the saved model and transformer ---
@st.cache_resource # Cache the model and transformer loading
def load_resources():
    try:
        poly_transformer = joblib.load('polynomial_features_transformer.pkl')
    except FileNotFoundError:
        st.error("Error: 'polynomial_features_transformer.pkl' not found. Please ensure it's in the same directory.")
        st.stop()

    try:
        loaded_gb_model = joblib.load('model.pkl')
    except FileNotFoundError:
        st.error("Error: 'model.pkl' not found. Please ensure it's in the same directory.")
        st.stop()

    return poly_transformer, loaded_gb_model

poly_transformer, loaded_gb_model = load_resources()

# --- 2. Define Optimal Threshold ---
OPTIMAL_THRESHOLD = 0.25

# --- 3. Streamlit App Layout ---
st.set_page_config(page_title="Heart Failure Risk Prediction", layout="centered")
st.title("Heart Failure Risk Prediction🩺")
st.markdown("--- ")
st.markdown("### Enter Patient's Clinical Parameters")
st.write("Adjust the values below to predict the risk of a heart disease event.")

# Input Fields Grouped for better UI
with st.form("prediction_form"):
    with st.expander("Demographics & Lifestyle", expanded=True):
        age = st.slider("Age (years)", 1, 100, 60, help="Patient's age in years.")
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male", help="Biological sex of the patient (0: Female, 1: Male).")
        smoking = st.selectbox("Smoking", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", help="Does the patient smoke? (1: Yes, 0: No).")

    with st.expander("Medical Conditions", expanded=True):
        anaemia = st.selectbox("Anaemia", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", help="Does the patient have anaemia? (1: Yes, 0: No).")
        diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", help="Does the patient have diabetes? (1: Yes, 0: No).")
        high_blood_pressure = st.selectbox("High Blood Pressure", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", help="Does the patient have high blood pressure? (1: Yes, 0: No).")

    with st.expander("Clinical Measurements", expanded=True):
        creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase (CPK) (mcg/L)", 0, 10000, 250, help="Level of the CPK enzyme in the blood. Higher levels indicate muscle damage.")
        ejection_fraction = st.slider("Ejection Fraction (%)", 10, 100, 40, help="Percentage of blood leaving the heart at each contraction. Lower values indicate heart failure.")
        platelets = st.number_input("Platelets (kiloplatelets/mL)", 0, 1000000, 250000, help="Platelets in the blood. Lower values indicate bleeding disorders.")
        serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", 0.0, 10.0, 1.2, step=0.1, format="%.1f", help="Level of serum creatinine in the blood. Higher values indicate kidney dysfunction.")
        serum_sodium = st.number_input("Serum Sodium (mEq/L)", 100, 150, 135, help="Level of serum sodium in the blood. Lower values can indicate heart problems.")

    st.markdown("--- ")
    submit_button = st.form_submit_button(" Predict Heart Disease ")

# --- 4. Prediction Logic ---
if submit_button:
    input_data = {
        'age': age,
        'anaemia': anaemia,
        'creatinine_phosphokinase': creatinine_phosphokinase,
        'diabetes': diabetes,
        'ejection_fraction': ejection_fraction,
        'high_blood_pressure': high_blood_pressure,
        'platelets': platelets,
        'serum_creatinine': serum_creatinine,
        'serum_sodium': serum_sodium,
        'sex': sex,
        'smoking': smoking
    }

    # Convert input to DataFrame
    new_raw_sample = pd.DataFrame([input_data])

    # Apply polynomial feature transformation
    try:
        # The poly_transformer was fit on data with the 11 original features
        original_feature_cols = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction',
                                 'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking']
        transformed_features = poly_transformer.transform(new_raw_sample[original_feature_cols])
    except Exception as e:
        st.error(f"An error occurred during feature transformation: {e}")
        st.stop()

    # Make prediction (get probabilities)
    try:
        prediction_proba = loaded_gb_model.predict_proba(transformed_features)[:, 1]
        # Apply the optimal threshold
        final_prediction = (prediction_proba >= OPTIMAL_THRESHOLD).astype(int)[0]

        st.subheader("Prediction Result:")
        if final_prediction == 1:
            st.error(f"## ⚠️ High Risk of Heart Disease Event!")
            st.write(f"Based on the input parameters and a classification threshold of {OPTIMAL_THRESHOLD}, the model predicts a high likelihood of a heart failure event.")
            st.info("It is strongly recommended to consult a medical professional for further evaluation and diagnosis.")
        else:
            st.success(f"## ✅ Low Risk of Heart Disease Event")
            st.write(f"Based on the input parameters and a classification threshold of {OPTIMAL_THRESHOLD}, the model predicts a low likelihood of a heart failure event.")
            st.info("While the risk is low, continuous monitoring and regular check-ups are always advisable for heart health.")

        st.markdown("--- ")
        st.warning("**Disclaimer:** This prediction is generated by an AI model and is for informational purposes only. It should not be considered medical advice or a substitute for professional medical consultation, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.")

    except Exception as e:
        st.error(f"An error occurred during model prediction: {e}")
# Team Crredits

with st.expander("ℹ️ About the Project & Team"):
    st.write("""
    This Heart Failure Risk Prediction tool uses a Gradient Boosting model 
    enhanced with Polynomial Features to identify clinical risk factors.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Project Lead**")
        st.write("[Albert Antony S](https://linkedin.com/in/albertantonys)")
    with col2:
        st.markdown("**Collaborators**")
        st.write("Jeevanesh K")
        st.write("Kumaran M")
        st.write("Elaichandiran S")
