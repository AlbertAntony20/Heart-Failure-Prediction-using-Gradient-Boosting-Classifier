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
        st.error("Error: 'polynomial_features_transformer.pkl' not found. Make sure it's in the same directory.")
        st.stop()
    
    try:
        loaded_gb_model = joblib.load('model.pkl')
    except FileNotFoundError:
        st.error("Error: 'model.pkl' not found. Make sure it's in the same directory.")
        st.stop()
        
    return poly_transformer, loaded_gb_model

poly_transformer, loaded_gb_model = load_resources()

# --- 2. Define Optimal Threshold and Constants ---
OPTIMAL_THRESHOLD = 0.25
# Define thresholds for targeted feature engineering (if used in app.py logic before poly transform)
# Based on previous analysis, our `best_gb_model` (from GridSearchCV) was trained on X_train_poly (from 11 original features).
# The targeted FE was NOT part of the `best_gb_model`'s training, it was part of the *later* ADASYN/Cost-Sensitive experiment.
# Therefore, the `app.py` here should ONLY apply polynomial transformation to the original 11 features.
# The comments about targeted FE in the previous app.py were for tracing previous notebook steps, but for this specific model, it's not applied here.

# --- 3. Streamlit App Layout ---
st.set_page_config(page_title="Heart Disease Event Prediction", layout="centered")
st.title("Heart Disease Event Prediction")
st.markdown("Enter patient's clinical parameters to predict the risk of a heart disease event.")

# --- 4. Input Fields ---
st.header("Patient Parameters")

with st.form("prediction_form"):
    age = st.slider("Age", 40, 100, 60)
    anaemia = st.selectbox("Anaemia", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase (CPK)", 0, 10000, 250)
    diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    ejection_fraction = st.slider("Ejection Fraction (%)", 10, 100, 40)
    high_blood_pressure = st.selectbox("High Blood Pressure", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    platelets = st.number_input("Platelets (kiloplatelets/mL)", 0, 1000000, 250000)
    serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", 0.0, 10.0, 1.2, step=0.1)
    serum_sodium = st.number_input("Serum Sodium (mEq/L)", 100, 150, 135)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    smoking = st.selectbox("Smoking", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    submit_button = st.form_submit_button("Predict Heart Disease Event")

# --- 5. Prediction Logic ---
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
        transformed_features = poly_transformer.transform(new_raw_sample)
    except Exception as e:
        st.error(f"Error applying polynomial transformation: {e}")
        st.stop()

    # Make prediction (get probabilities)
    try:
        prediction_proba = loaded_gb_model.predict_proba(transformed_features)[:, 1]
        # Apply the optimal threshold
        final_prediction = (prediction_proba >= OPTIMAL_THRESHOLD).astype(int)[0]
        
        st.subheader("Prediction Result:")
        if final_prediction == 1:
            st.error(f"High risk of Heart Disease Event! (Probability: {prediction_proba[0]:.2f})")
            st.write(f"*Prediction based on a threshold of {OPTIMAL_THRESHOLD}.*")
        else:
            st.success(f"Low risk of Heart Disease Event. (Probability: {prediction_proba[0]:.2f})")
            st.write(f"*Prediction based on a threshold of {OPTIMAL_THRESHOLD}.*")
            
        st.markdown("--- ")
        st.info("Disclaimer: This prediction is for informational purposes only and should not be used as a substitute for professional medical advice.")

    except Exception as e:
        st.error(f"Error during model prediction: {e}")