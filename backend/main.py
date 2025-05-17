from fastapi import FastAPI
import joblib
import numpy as np
import uvicorn
from pydantic import BaseModel

# Load the model and scaler
model = joblib.load('models/heart_failure_model.pkl')
scaler = joblib.load('models/scaler.pkl')

app = FastAPI()

class InputData(BaseModel):
    age: int
    anaemia: int
    creatinine_phosphokinase: int
    diabetes: int
    ejection_fraction: int
    high_blood_pressure: int
    platelets: float
    serum_creatinine: float
    serum_sodium: int
    sex: int
    smoking: int
    time: int

@app.post("/predict")
def predict(data: InputData):
    # Convert input data to numpy array and reshape for prediction
    features = np.array([
        data.age,
        data.anaemia,
        data.creatinine_phosphokinase,
        data.diabetes,
        data.ejection_fraction,
        data.high_blood_pressure,
        data.platelets,
        data.serum_creatinine,
        data.serum_sodium,
        data.sex,
        data.smoking,
        data.time
    ]).reshape(1, -1)
    
    # Scale the features
    scaled_features = scaler.transform(features)
    
    # Get prediction and probability
    prediction = model.predict(scaled_features)
    probability = model.predict_proba(scaled_features)
    
    # Determine risk level based on probability
    risk_probability = probability[0][1] * 100
    if risk_probability >= 70:
        risk_level = "High Risk"
    elif risk_probability >= 40:
        risk_level = "Medium Risk"
    else:
        risk_level = "Low Risk"
    
    return {
        "prediction": "Heart Failure Risk Detected" if prediction[0] == 1 else "No Heart Failure Risk",
        "risk_level": risk_level,
        "probability": f"{risk_probability:.2f}%",
        "details": {
            "age": data.age,
            "ejection_fraction": data.ejection_fraction,
            "serum_creatinine": data.serum_creatinine,
            "serum_sodium": data.serum_sodium
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
