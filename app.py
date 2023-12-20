import streamlit as st
import pickle
import pandas as pd

# Load the model
with open("model.sav", "rb") as file:
    model = pickle.load(file)

# Collect input parameters from the user
age = st.number_input("Age", min_value=0, max_value=120, step=1)
systolic_bp = st.number_input("Systolic Blood Pressure", min_value=0, max_value=300, step=1)
diastolic_bp = st.number_input("Diastolic Blood Pressure", min_value=0, max_value=200, step=1)
bs = st.number_input("Blood Sugar", min_value=0, max_value=300, step=1)
body_temp = st.number_input("Body Temperature", min_value=30.0, max_value=43.0, step=0.1)
heart_rate = st.number_input("Heart Rate", min_value=0, max_value=200, step=1)

# Create a DataFrame with the user input
input_data = pd.DataFrame({
    'Age': [age],
    'SystolicBP': [systolic_bp],
    'DiastolicBP': [diastolic_bp],
    'BS': [bs],
    'BodyTemp': [body_temp],
    'HeartRate': [heart_rate]
})

# Use the same scaler object from the model
scaler = model.named_steps['scaler']  # Adjust based on the actual name used in your pipeline

# Scale the input features
scaled_input_data = scaler.transform(input_data)

# Make predictions using the model
prediction = model.predict(scaled_input_data)[0]

# Rest of your code for displaying the prediction
