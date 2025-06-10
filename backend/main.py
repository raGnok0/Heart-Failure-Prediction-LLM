from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import uvicorn
from pydantic import BaseModel
from typing import Optional
from model_comparison import model_comparison

app = FastAPI(
    title="Heart Failure Prediction API",
    description="Multi-model heart failure prediction system with comparison capabilities",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the legacy model and scaler for backward compatibility
try:
    legacy_model = joblib.load('models/heart_failure_model.pkl')
    legacy_scaler = joblib.load('models/scaler.pkl')
except:
    legacy_model = None
    legacy_scaler = None

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

def prepare_features(data: InputData) -> np.ndarray:
    """Convert input data to numpy array"""
    return np.array([
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

@app.post("/predict")
def predict_legacy(data: InputData):
    """Legacy prediction endpoint for backward compatibility"""
    if not legacy_model or not legacy_scaler:
        raise HTTPException(status_code=503, detail="Legacy model not available")
    
    features = prepare_features(data)
    scaled_features = legacy_scaler.transform(features)
    
    prediction = legacy_model.predict(scaled_features)
    probability = legacy_model.predict_proba(scaled_features)
    
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
        "model_used": "xgboost_legacy",
        "details": {
            "age": data.age,
            "ejection_fraction": data.ejection_fraction,
            "serum_creatinine": data.serum_creatinine,
            "serum_sodium": data.serum_sodium
        }
    }

@app.post("/predict/single/{model_name}")
def predict_single_model(model_name: str, data: InputData):
    """Make prediction using a specific model"""
    try:
        features = prepare_features(data)
        result = model_comparison.predict_single_model(model_name, features)
        
        # Add input details
        result["details"] = {
            "age": data.age,
            "ejection_fraction": data.ejection_fraction,
            "serum_creatinine": data.serum_creatinine,
            "serum_sodium": data.serum_sodium
        }
        
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/compare")
def predict_compare_all(data: InputData):
    """Make predictions using all available models and compare results"""
    try:
        features = prepare_features(data)
        result = model_comparison.predict_all_models(features)
        
        # Add input details
        result["input_details"] = {
            "age": data.age,
            "ejection_fraction": data.ejection_fraction,
            "serum_creatinine": data.serum_creatinine,
            "serum_sodium": data.serum_sodium,
            "all_features": {
                "age": data.age,
                "anaemia": data.anaemia,
                "creatinine_phosphokinase": data.creatinine_phosphokinase,
                "diabetes": data.diabetes,
                "ejection_fraction": data.ejection_fraction,
                "high_blood_pressure": data.high_blood_pressure,
                "platelets": data.platelets,
                "serum_creatinine": data.serum_creatinine,
                "serum_sodium": data.serum_sodium,
                "sex": data.sex,
                "smoking": data.smoking,
                "time": data.time
            }
        }
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison error: {str(e)}")

@app.get("/models/available")
def get_available_models():
    """Get list of available models"""
    return {
        "available_models": model_comparison.get_available_models(),
        "total_models": len(model_comparison.get_available_models())
    }

@app.get("/models/performance")
def get_model_performance():
    """Get performance metrics for all models"""
    return model_comparison.get_model_performance()

@app.get("/models/performance/{model_name}")
def get_single_model_performance(model_name: str):
    """Get performance metrics for a specific model"""
    performance_data = model_comparison.get_model_performance()
    
    if "error" in performance_data:
        raise HTTPException(status_code=503, detail=performance_data["error"])
    
    if model_name not in performance_data["performance_metrics"]:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    return {
        "model": model_name,
        "performance": performance_data["performance_metrics"][model_name],
        "ranking_position": {
            metric: next((i+1 for i, item in enumerate(rankings) if item["model"] == model_name), None)
            for metric, rankings in performance_data["rankings"].items()
        }
    }

@app.get("/models/feature-importance")
def get_all_feature_importance():
    """Get feature importance for all models"""
    return model_comparison.get_feature_importance()

@app.get("/models/feature-importance/{model_name}")
def get_model_feature_importance(model_name: str):
    """Get feature importance for a specific model"""
    result = model_comparison.get_feature_importance(model_name)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

@app.get("/models/info")
def get_models_info():
    """Get general information about all models"""
    return model_comparison.get_model_info()

@app.get("/")
def root():
    """API root endpoint with information"""
    return {
        "message": "Heart Failure Prediction API",
        "version": "2.0.0",
        "description": "Multi-model heart failure prediction system",
        "available_endpoints": {
            "prediction": {
                "/predict": "Legacy single model prediction",
                "/predict/single/{model_name}": "Predict with specific model",
                "/predict/compare": "Compare predictions from all models"
            },
            "models": {
                "/models/available": "List available models",
                "/models/performance": "Get all model performance metrics",
                "/models/performance/{model_name}": "Get specific model performance",
                "/models/feature-importance": "Get feature importance for all models",
                "/models/feature-importance/{model_name}": "Get feature importance for specific model",
                "/models/info": "Get general model information"
            }
        },
        "models_loaded": len(model_comparison.get_available_models()),
        "features_required": model_comparison.feature_names
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
