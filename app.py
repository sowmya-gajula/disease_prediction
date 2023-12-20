import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

with open("model.pkl", "rb") as file:
    model = pickle.load(file)


scaler = MinMaxScaler()

st.title("Health Prediction App")
st.subheader("Enter Health Parameters:")

age = st.number_input("Age", min_value=0, max_value=120, step=1)
systolic_bp = st.number_input("Systolic Blood Pressure", min_value=0, max_value=300, step=1)
diastolic_bp = st.number_input("Diastolic Blood Pressure", min_value=0, max_value=200, step=1)
bs = st.number_input("Blood Sugar", min_value=0, max_value=300, step=1)
body_temp = st.number_input("Body Temperature", min_value=30.0, max_value=43.0, step=0.1)
heart_rate = st.number_input("Heart Rate", min_value=0, max_value=200, step=1)

if st.button("Predict"):
    input_data = pd.DataFrame({
        'Age': [age],
        'SystolicBP': [systolic_bp],
        'DiastolicBP': [diastolic_bp],
        'BS': [bs],
        'BodyTemp': [body_temp],
        'HeartRate': [heart_rate]
    })

    # Scale the input features
    scaled_input_data = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_input_data)[0]

    # Define threshold values for low, medium, and high
    low_threshold = 0.33
    high_threshold = 0.66

    # Map the prediction to discrete categories
    if prediction <= low_threshold:
        category = "Low"
    elif low_threshold < prediction <= high_threshold:
        category = "Medium"
    else:
        category = "High"

    st.header("Prediction:")
    st.subheader(f"The predicted health category is: {category}")
