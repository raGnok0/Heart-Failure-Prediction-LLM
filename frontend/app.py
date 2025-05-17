import streamlit as st
import requests

# Set page config
st.set_page_config(page_title="Heart Failure Prediction", page_icon="💓", layout="wide")

st.title("💓 Heart Failure Prediction")
st.write("Enter patient details below to predict heart failure risk.")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Patient Information")
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    smoking = st.selectbox("Smoking Status", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    diabetes = st.selectbox("Diabetes", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    high_blood_pressure = st.selectbox("High Blood Pressure", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    anaemia = st.selectbox("Anaemia", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

with col2:
    st.subheader("Clinical Measurements")
    ejection_fraction = st.number_input("Ejection Fraction (%)", min_value=0, max_value=100, value=40)
    serum_creatinine = st.number_input("Serum Creatinine Level", min_value=0.1, max_value=10.0, value=1.0)
    serum_sodium = st.number_input("Serum Sodium Level", min_value=100, max_value=150, value=135)
    platelets = st.number_input("Platelets Count", min_value=0, max_value=1000000, value=250000)
    creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase", min_value=0, max_value=10000, value=500)
    time = st.number_input("Follow-up Period (days)", min_value=0, max_value=300, value=10)

# Prepare input data
input_data = {
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
}

# Button to send API request
if st.button("Predict Heart Failure Risk"):
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=input_data)
        result = response.json()
        
        # Display results in a nice format
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        # Create three columns for results
        pred_col, risk_col, prob_col = st.columns(3)
        
        with pred_col:
            st.metric("Prediction", result["prediction"])
        
        with risk_col:
            st.metric("Risk Level", result["risk_level"])
        
        with prob_col:
            st.metric("Probability", result["probability"])
        
        # Display detailed information
        st.markdown("### 📋 Clinical Details")
        details = result["details"]
        st.write(f"- Age: {details['age']} years")
        st.write(f"- Ejection Fraction: {details['ejection_fraction']}%")
        st.write(f"- Serum Creatinine: {details['serum_creatinine']} mg/dL")
        st.write(f"- Serum Sodium: {details['serum_sodium']} mEq/L")
        
        # Add color-coded risk indicator
        risk_color = {
            "High Risk": "🔴",
            "Medium Risk": "🟡",
            "Low Risk": "🟢"
        }
        st.markdown(f"### {risk_color[result['risk_level']]} Risk Assessment")
        
    except requests.exceptions.RequestException:
        st.error("❌ Could not connect to the backend API. Please make sure the backend server is running.")
