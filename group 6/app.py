import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Group 6 Diabetes Prediction App", page_icon=":guardsman:", layout="wide")

# custom css
st.markdown(
    """
    <style>
    .main {
        padding: 0rem 1rem;
        background-color: #f5f5f5;
    }
    .stAlert {
        background-color: #ffcccc;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .h1 {
        color: #333333;
        padding-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True)

# Load the trained model and scaler
@st.cache_resource
def load_model():
    try:
        model = joblib.load('diabetes_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        return None, None

    #header
st.title("Group 6 Diabetes Prediction App")
st.markdown("This app predicts whether a person is diabetic or not based on their health parameters.")

#load model and scaler
model, scaler = load_model()

if model is None or scaler is None:
    st.error("Model or Scaler not found.")


    #sidebar for user input
st.sidebar.title("User Input Info")
st.sidebar.subheader("Enter Details:")
age = st.sidebar.slider("Age", 1, 100, 25)
pregnancies = st.sidebar.slider("Number of Pregnancies", 0, 20, 0)

st.sidebar.subheader("Health Tests Results:")
glucose = st.sidebar.slider("Glucose Level", 0, 200, 100)
blood_pressure = st.sidebar.slider("Blood Pressure", 0, 120, 80)
skin_thickness = st.sidebar.slider("Skin Thickness", 0, 99, 20)
insulin = st.sidebar.slider("Insulin Level", 0, 846, 100)
bmi = st.sidebar.slider("BMI", 0.0, 67.1, 25.0)
diabetes_pedigree = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.42, 0.5)

#predict button
st.sidebar.subheader("Prediction:")
predict_btn = st.sidebar.button("Predict", type="primary", use_container_width=True)

#main content

if predict_btn:
    if model is not None and scaler is not None:
        input_data = (pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age)
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        std_data = scaler.transform(input_data_reshaped)
        prediction = model.predict(std_data)

        try:
            probabilities = model.predict_proba(std_data)[0]
            prob_negative = probabilities[0] * 100
            prob_positive = probabilities[1] * 100
        except:
            prob_positive = 100 if prediction == 1 else 0
            prob_negative = 100 - prob_positive

    #display results
    st.subheader("Prediction Result:")

    col1, col2 = st.columns([2, 1])

    with col1:
        if prediction == 0:
            if prob_positive < 30:
                st.success(f"Low risk of diabetes with a probability of {prob_positive:.2f}% - Not Diabetic.")
            else:
                st.warning(f"Moderate risk of diabetes with a probability of {prob_positive:.2f}% - Not Diabetic.")

        else:
            if prob_positive > 70:
                st.error(f"High risk of diabetes with a probability of {prob_positive:.2f}% - Diabetic.")
            else:
                st.warning(f"Moderate risk of diabetes with a probability of {prob_positive:.2f}% - Diabetic.")

    #probability Breakdown

    st.subheader("Probability Breakdown:")
    pcol1, col2 = st.columns(2)
    pcol1.metric("Probability of Not Diabetic", f"{prob_negative:.2f}%")
    pcol1.metric("Probability of Diabetic", f"{prob_positive:.2f}%")

    with col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_positive,
            title={'text': "Diabetic Probability"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "red"},
                   'steps': [
                       {'range': [0, 30], 'color': "green"},
                       {'range': [30, 70], 'color': "yellow"},
                       {'range': [70, 100], 'color': "red"}],
                       'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': prob_positive}}))
        fig.update_layout(height=400, width=400, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

    #Risk Assessment
    st.subheader("Risk Assessment:")

    risk_factors = []
    positive_factors = []

    if glucose > 125:
        risk_factors.append("High Glucose Level")
    elif glucose < 100:
        positive_factors.append("Low Glucose Level")

    if blood_pressure > 90:
        risk_factors.append("High Blood Pressure")
    elif 60<= blood_pressure <= 80:
        positive_factors.append("Normal Blood Pressure")

    if bmi > 30:
        risk_factors.append("High BMI")
    elif 18.5 <= bmi <= 24.5:
        positive_factors.append("Normal BMI")


    if age > 45:
        risk_factors.append("Older Age")
    elif age < 30:
        positive_factors.append("Younger Age")

    if risk_factors:
        st.warning("Risk Factors Identified:")
        for factor in risk_factors:
            st.write(f"- {factor}")
    else:
        st.success("No significant risk factors identified.")

# Recommendations
    st.subheader("Recommendations:")
    if prediction == 1:
        st.write("Based on the prediction, it is recommended to consult a healthcare professional for further evaluation and management of diabetes.")
    else:
        st.write("Maintain a healthy lifestyle, regular exercise, and balanced diet to prevent diabetes.") 

# Disclaimer
    st.subheader("Disclaimer:")
    st.info("This prediction is based on a machine learning model and should" \
    " not be considered as a definitive medical diagnosis. Please consult a " \
    "healthcare professional for accurate assessment and advice.")

else:
    st.write("Please enter your details in the sidebar and click 'Predict' to see the results.")
    