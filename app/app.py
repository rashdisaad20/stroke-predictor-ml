import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os 




# giving exact diractory 
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 

# Project ROOT 
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# joining with file paths 

model_path = os.path.join(PROJECT_ROOT, 'notebooks','model.pkl')
scalar_path = os.path.join(PROJECT_ROOT,'notebooks','scalar.pkl')
# 1. Load the "Brains" of your project
if os.path.exists(model_path):
    model = joblib.load(model_path)
    scalar = joblib.load(scalar_path)
    


# 2. Setup the Page Header
st.set_page_config(page_title="Stroke Risk Predictor", layout="centered")
st.title("🏥 Stroke Risk Prediction System")
st.markdown("---")
st.write("Enter the patient's data below to calculate the probability of a stroke.")

# 3. Create the Input Form (Sliders & Inputs)
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 1, 100, 45)
        glucose = st.number_input("Glucose Level (mg/dL)", 50.0, 300.0, 100.0)
        cigs = st.slider("Cigarettes Per Day", 0, 70, 0)
        diabetes = st.selectbox("Diabetes?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    with col2:
        sysBP = st.number_input("Systolic BP", 80, 220, 120)
        diaBP = st.number_input("Diastolic BP", 40, 120, 80)
        # Feature Engineering: We calculate this exactly like your EDA
        pulse_pressure = sysBP - diaBP
        age_glucose_risk = age * glucose

# 4. The Prediction Button
if st.button("Calculate Risk", use_container_width=True):
    
    # Create the input dataframe (MUST match the order of your X_train)
    # The order: age, cigsPerDay, sysBP, diaBP, glucose, diabetes, pulse_pressure, age_glucose_risk
    features = pd.DataFrame([[age, cigs, sysBP, diaBP, glucose, diabetes, pulse_pressure, age_glucose_risk]],
                           columns=['age', 'cigsPerDay', 'sysBP', 'diaBP', 'glucose', 'diabetes', 'pulse_pressure', 'age_glucose_risk'])
    
    # Scale the data using your saved scaler
    scaled_features = scalar.transform(features)
    
    # Get the results
    prediction = model.predict(scaled_features)
    probability = model.predict_proba(scaled_features)[0][1]
    
    # 5. Display the Result
    st.divider()
    if prediction[0] == 1:
        st.error(f"⚠️ HIGH RISK: {probability*100:.1f}% probability of stroke.")
    else:
        st.success(f"✅ LOW RISK: {probability*100:.1f}% probability of stroke.")