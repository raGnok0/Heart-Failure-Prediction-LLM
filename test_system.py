import requests
import json

def test_api_endpoints():
    """Test all API endpoints to ensure the multi-model system is working"""
    base_url = "http://127.0.0.1:8000"
    
    # Sample patient data for testing
    test_data = {
        "age": 75,
        "anaemia": 0,
        "creatinine_phosphokinase": 582,
        "diabetes": 0,
        "ejection_fraction": 20,
        "high_blood_pressure": 1,
        "platelets": 265000,
        "serum_creatinine": 1.9,
        "serum_sodium": 130,
        "sex": 1,
        "smoking": 0,
        "time": 4
    }
    
    print("🧪 Testing Heart Failure Prediction Multi-Model System")
    print("=" * 60)
    
    # Test 1: Root endpoint
    print("\n1. Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Root endpoint working")
            print(f"   Version: {data.get('version', 'N/A')}")
            print(f"   Models loaded: {data.get('models_loaded', 0)}")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
    
    # Test 2: Available models
    print("\n2. Testing available models endpoint...")
    try:
        response = requests.get(f"{base_url}/models/available")
        if response.status_code == 200:
            data = response.json()
            models = data.get("available_models", [])
            print(f"✅ Available models: {models}")
            print(f"   Total models: {data.get('total_models', 0)}")
        else:
            print(f"❌ Available models failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Available models error: {e}")
    
    # Test 3: Model performance
    print("\n3. Testing model performance endpoint...")
    try:
        response = requests.get(f"{base_url}/models/performance")
        if response.status_code == 200:
            data = response.json()
            if "error" not in data:
                print(f"✅ Performance data available")
                print(f"   Best model: {data.get('best_overall_model', 'N/A')}")
                
                # Show performance summary
                if "performance_metrics" in data:
                    print("   Performance Summary:")
                    for model, metrics in data["performance_metrics"].items():
                        accuracy = metrics.get("accuracy", 0)
                        f1 = metrics.get("f1_score", 0)
                        print(f"     {model}: Accuracy={accuracy:.3f}, F1={f1:.3f}")
            else:
                print(f"⚠️ Performance data not available: {data['error']}")
        else:
            print(f"❌ Performance endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Performance endpoint error: {e}")
    
    # Test 4: Legacy prediction
    print("\n4. Testing legacy prediction endpoint...")
    try:
        response = requests.post(f"{base_url}/predict", json=test_data)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Legacy prediction working")
            print(f"   Prediction: {result.get('prediction', 'N/A')}")
            print(f"   Risk Level: {result.get('risk_level', 'N/A')}")
            print(f"   Probability: {result.get('probability', 'N/A')}")
        else:
            print(f"❌ Legacy prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Legacy prediction error: {e}")
    
    # Test 5: Single model predictions
    print("\n5. Testing single model predictions...")
    models_to_test = ["xgboost", "random_forest", "logistic_regression"]
    
    for model in models_to_test:
        try:
            response = requests.post(f"{base_url}/predict/single/{model}", json=test_data)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ {model}: {result.get('prediction', 'N/A')} ({result.get('probability', 'N/A')})")
            else:
                print(f"❌ {model} failed: {response.status_code}")
        except Exception as e:
            print(f"❌ {model} error: {e}")
    
    # Test 6: Multi-model comparison
    print("\n6. Testing multi-model comparison...")
    try:
        response = requests.post(f"{base_url}/predict/compare", json=test_data)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Multi-model comparison working")
            
            # Ensemble results
            ensemble = result.get("ensemble_prediction", {})
            print(f"   Ensemble Prediction: {ensemble.get('prediction', 'N/A')}")
            print(f"   Average Probability: {ensemble.get('average_probability', 'N/A')}")
            print(f"   Consensus Risk: {ensemble.get('consensus_risk', 'N/A')}")
            print(f"   Consensus Strength: {ensemble.get('consensus_strength', 'N/A')}")
            
            # Individual model results
            individual = result.get("individual_predictions", {})
            print(f"   Individual Results:")
            for model, pred in individual.items():
                if "error" not in pred:
                    print(f"     {model}: {pred.get('risk_level', 'N/A')} ({pred.get('probability', 'N/A')})")
                else:
                    print(f"     {model}: Error - {pred.get('error', 'Unknown')}")
            
            print(f"   Models Used: {result.get('model_count', 0)}")
            print(f"   Successful Predictions: {result.get('successful_predictions', 0)}")
            
        else:
            print(f"❌ Multi-model comparison failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Multi-model comparison error: {e}")
    
    # Test 7: Feature importance
    print("\n7. Testing feature importance endpoint...")
    try:
        response = requests.get(f"{base_url}/models/feature-importance")
        if response.status_code == 200:
            data = response.json()
            if "all_models" in data:
                print(f"✅ Feature importance available")
                available_models = data.get("available_models", [])
                print(f"   Models with feature importance: {available_models}")
                
                # Show top features for first model
                if available_models:
                    first_model = available_models[0]
                    model_features = data["all_models"].get(first_model, {})
                    if model_features:
                        # Sort by importance
                        sorted_features = sorted(model_features.items(), key=lambda x: x[1], reverse=True)
                        print(f"   Top 5 features for {first_model}:")
                        for feature, importance in sorted_features[:5]:
                            print(f"     {feature}: {importance:.4f}")
            else:
                print(f"⚠️ Feature importance not available")
        else:
            print(f"❌ Feature importance failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Feature importance error: {e}")
    
    print("\n" + "=" * 60)
    print("🏁 Testing completed!")
    print("\n📋 Summary:")
    print("- If all tests show ✅, your multi-model system is working correctly")
    print("- If you see ❌, check that the backend server is running on port 8000")
    print("- If you see ⚠️, some features may need additional setup")
    print("\n🚀 To start the system:")
    print("1. Backend: cd backend && python main.py")
    print("2. Frontend: cd frontend && streamlit run multi_model_app.py")

if __name__ == "__main__":
    test_api_endpoints()
