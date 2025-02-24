import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load trained model, scaler, and feature names
model = joblib.load("logistic_regression.joblib")
scaler = joblib.load("scaler.joblib")
feature_names = joblib.load("feature_names.joblib")

# Preprocessing functions
def chestpain1(chestpain):
    mapping = {"ATA": [1, 0, 0], "NAP": [0, 1, 0], "TA": [0, 0, 1], "ASY": [0, 0, 0]}
    return mapping.get(chestpain.upper(), [0, 0, 0])

def slope1(slope):
    mapping = {"UP": [0, 1], "FLAT": [1, 0], "DOWN": [0, 0]}
    return mapping.get(slope.upper(), [0, 0])

def restingecg1(resting_ecg):
    mapping = {"NORMAL": [1, 0], "ST": [0, 1], "LVH": [0, 0]}
    return mapping.get(resting_ecg.upper(), [0, 0])

def fastingbs1(fastingbs):
    return [1] if fastingbs.lower() == 'high' else [0]

def agena(exercise):
    return [1] if exercise.lower() == 'yes' else [0]

def gender1(gender):
    return [1] if gender.lower() == 'male' else [0]

def preprocess_input(age, gender, chestpain, restingbp, cholestrol, fastingbs, restingecg, maxhr, exercise, oldpeak, slope):
    """Convert user inputs into the correct feature format."""
    
    # Convert categorical inputs to one-hot encoding
    categorical_data = gender1(gender) + chestpain1(chestpain) + restingecg1(restingecg) + agena(exercise) + slope1(slope) + fastingbs1(fastingbs)
    
    # Numerical features
    numerical_features = [age, restingbp, cholestrol, maxhr, oldpeak]

    # Combine numerical and categorical features
    full_features = numerical_features + categorical_data
    
    # Convert to DataFrame to match feature order
    input_df = pd.DataFrame([full_features], columns=feature_names)
    
    return input_df

def predict(age, gender, chestpain, restingbp, cholestrol, fastingbs, restingecg, maxhr, exercise, oldpeak, slope):
    # Process input
    input_df = preprocess_input(age, gender, chestpain, restingbp, cholestrol, fastingbs, restingecg, maxhr, exercise, oldpeak, slope)

    # Standardize using trained scaler
    standardized_features = scaler.transform(input_df)

    # Get prediction probability and raw prediction
    probability = model.predict_proba(standardized_features)[0][1]  # Probability of heart disease
    raw_prediction = model.predict(standardized_features)[0]  # Raw 0/1 prediction

    # Print debug information
    print(f"🔎 Prediction Probability: {probability:.4f}")
    print(f"🔎 Raw Model Prediction: {raw_prediction}")

    # Adjust threshold if necessary
    threshold = 0.6  # Try adjusting this
    result = 1 if probability >= threshold else 0

    return "⚠️ You may be at risk of heart disease. Consult a doctor." if result == 1 else "✅ You are not at risk of heart disease."
def main():
    st.title("💖 Heart Disease Prediction")

    # Input fields
    age = st.number_input("🧓 AGE", min_value=1, max_value=100, value=50)
    gender = st.radio('🧑 Select Gender', ("Male", "Female"), horizontal=True)
    chestpain = st.selectbox("💔 Choose Your Chest Pain Type", ('ATA', 'NAP', 'TA', 'ASY'))
    restingbp = st.slider("🩸 Resting Blood Pressure", min_value=10, max_value=200, value=120)
    cholestrol = st.number_input("🩺 Cholesterol Level", min_value=10, max_value=500, value=200)
    fastingbs = st.radio("🍽️ Fasting Blood Sugar", ('Low', 'High'), horizontal=True)
    restingecg = st.selectbox("📊 Resting ECG Type", ("Normal", "LVH", "ST"))
    maxhr = st.slider("❤️ Max Heart Rate", min_value=35, max_value=250, value=150)
    exercise = st.radio('🏃 Exercise-Induced Angina?', ("Yes", "No"), horizontal=True)
    oldpeak = st.number_input("📉 Old Peak Value", min_value=-6.0, max_value=7.0, value=1.0)
    slope = st.selectbox("📈 Slope of ST Segment", ('UP', 'DOWN', 'FLAT'))
    
    # Predict button
    if st.button("🔍 Predict"):
        result = predict(age, gender, chestpain, restingbp, cholestrol, fastingbs, restingecg, maxhr, exercise, oldpeak, slope)
        st.subheader(result)

if __name__ == '__main__':
    main()
