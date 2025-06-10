import streamlit as st
import requests
import pandas as pd

# Set page config
st.set_page_config(
    page_title="Heart Failure Prediction - Multi-Model System", 
    page_icon="💓", 
    layout="wide"
)

st.title("💓 Heart Failure Prediction - Multi-Model System")
st.write("Advanced heart failure prediction using multiple machine learning models with comparison capabilities.")

# Sidebar for model selection and information
with st.sidebar:
    st.header("🔧 Model Configuration")
    
    # Get available models
    try:
        models_response = requests.get("http://127.0.0.1:8000/models/available")
        if models_response.status_code == 200:
            available_models = models_response.json()["available_models"]
            st.success(f"✅ {len(available_models)} models loaded")
            
            # Model selection
            prediction_mode = st.radio(
                "Prediction Mode:",
                ["Single Model", "Compare All Models", "Model Performance"]
            )
            
            if prediction_mode == "Single Model":
                selected_model = st.selectbox("Select Model:", available_models)
            
        else:
            st.error("❌ Could not load model information")
            available_models = []
            prediction_mode = "Single Model"
            selected_model = "xgboost"
    except:
        st.error("❌ Backend not available")
        available_models = ["xgboost"]
        prediction_mode = "Single Model"
        selected_model = "xgboost"

# Main content area
if prediction_mode == "Model Performance":
    st.header("📊 Model Performance Dashboard")
    
    try:
        # Get performance data
        perf_response = requests.get("http://127.0.0.1:8000/models/performance")
        if perf_response.status_code == 200:
            perf_data = perf_response.json()
            
            if "error" not in perf_data:
                # Performance metrics table
                st.subheader("📈 Performance Metrics Comparison")
                
                metrics_df = pd.DataFrame(perf_data["performance_metrics"]).T
                metrics_df = metrics_df.drop("feature_importance", axis=1, errors="ignore")
                
                # Format the dataframe for better display
                formatted_df = metrics_df.round(4)
                st.dataframe(formatted_df, use_container_width=True)
                
                # Best model highlight
                best_model = perf_data.get("best_overall_model", "N/A")
                st.success(f"🏆 **Best Overall Model (F1-Score):** {best_model}")
                
                # Performance summary
                st.subheader("📊 Performance Summary")
                for model, metrics in perf_data["performance_metrics"].items():
                    with st.expander(f"{model.replace('_', ' ').title()} Performance"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
                            st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
                        with col2:
                            st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
                            st.metric("F1-Score", f"{metrics.get('f1_score', 0):.3f}")
                        with col3:
                            st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
                            st.metric("CV Accuracy", f"{metrics.get('cross_val_accuracy', 0):.3f}")
                
                # Feature importance
                st.subheader("🎯 Feature Importance Analysis")
                
                # Get feature importance data
                feat_response = requests.get("http://127.0.0.1:8000/models/feature-importance")
                if feat_response.status_code == 200:
                    feat_data = feat_response.json()
                    
                    if "all_models" in feat_data:
                        for model in feat_data["available_models"]:
                            if model in feat_data["all_models"]:
                                features = feat_data["all_models"][model]
                                if features:
                                    with st.expander(f"Feature Importance - {model.replace('_', ' ').title()}"):
                                        # Sort features by importance
                                        sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)
                                        
                                        # Create a simple bar chart using st.bar_chart
                                        feature_df = pd.DataFrame(sorted_features[:10], columns=['Feature', 'Importance'])
                                        feature_df = feature_df.set_index('Feature')
                                        st.bar_chart(feature_df)
                                        
                                        # Show top 5 features as text
                                        st.write("**Top 5 Features:**")
                                        for i, (feature, importance) in enumerate(sorted_features[:5], 1):
                                            st.write(f"{i}. {feature}: {importance:.4f}")
            else:
                st.error("Performance data not available")
        else:
            st.error("Could not fetch performance data")
    except Exception as e:
        st.error(f"Error loading performance data: {str(e)}")

else:
    # Input form for prediction
    st.header("📝 Patient Information")
    
    # Create input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Patient Demographics")
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
        smoking = st.selectbox("Smoking Status", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        diabetes = st.selectbox("Diabetes", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        high_blood_pressure = st.selectbox("High Blood Pressure", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        anaemia = st.selectbox("Anaemia", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    with col2:
        st.subheader("🔬 Clinical Measurements")
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
    
    # Prediction button
    predict_button = st.button("🔮 Predict Heart Failure Risk", type="primary", use_container_width=True)
    
    if predict_button:
        try:
            if prediction_mode == "Single Model":
                # Single model prediction
                response = requests.post(f"http://127.0.0.1:8000/predict/single/{selected_model}", json=input_data)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.markdown("---")
                    st.subheader(f"📊 Prediction Results - {result['model_name'].upper()}")
                    
                    # Display results
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Prediction", result["prediction"])
                    with col2:
                        st.metric("Risk Level", result["risk_level"])
                    with col3:
                        st.metric("Probability", result["probability"])
                    with col4:
                        st.metric("Confidence", result["confidence"])
                    
                    # Risk indicator
                    risk_colors = {
                        "High Risk": "🔴",
                        "Medium Risk": "🟡", 
                        "Low Risk": "🟢"
                    }
                    st.markdown(f"### {risk_colors.get(result['risk_level'], '⚪')} Risk Assessment: {result['risk_level']}")
                    
                else:
                    st.error(f"Prediction failed: {response.text}")
            
            elif prediction_mode == "Compare All Models":
                # Multi-model comparison
                response = requests.post("http://127.0.0.1:8000/predict/compare", json=input_data)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.markdown("---")
                    st.subheader("🔍 Multi-Model Comparison Results")
                    
                    # Ensemble prediction
                    ensemble = result["ensemble_prediction"]
                    st.markdown("### 🎯 Ensemble Prediction")
                    
                    ens_col1, ens_col2, ens_col3, ens_col4 = st.columns(4)
                    with ens_col1:
                        st.metric("Ensemble Prediction", ensemble["prediction"])
                    with ens_col2:
                        st.metric("Risk Level", ensemble["risk_level"])
                    with ens_col3:
                        st.metric("Average Probability", ensemble["average_probability"])
                    with ens_col4:
                        st.metric("Consensus Strength", ensemble["consensus_strength"])
                    
                    # Individual model results
                    st.markdown("### 📋 Individual Model Results")
                    
                    individual_results = []
                    for model_name, pred in result["individual_predictions"].items():
                        if "error" not in pred:
                            individual_results.append({
                                "Model": model_name.replace("_", " ").title(),
                                "Prediction": pred["prediction"],
                                "Risk Level": pred["risk_level"],
                                "Probability": pred["probability"],
                                "Confidence": pred["confidence"]
                            })
                    
                    if individual_results:
                        results_df = pd.DataFrame(individual_results)
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Simple visualization using st.bar_chart
                        st.markdown("### 📊 Risk Probability Comparison")
                        prob_data = {}
                        for r in individual_results:
                            prob_value = float(r["Probability"].replace("%", ""))
                            prob_data[r["Model"]] = prob_value
                        
                        prob_df = pd.DataFrame(list(prob_data.items()), columns=['Model', 'Risk Probability (%)'])
                        prob_df = prob_df.set_index('Model')
                        st.bar_chart(prob_df)
                        
                        # Add threshold line info
                        st.info("📏 **Threshold**: 50% - Above this indicates heart failure risk")
                    
                    # Summary statistics
                    st.markdown("### 📈 Prediction Summary")
                    summary_col1, summary_col2, summary_col3 = st.columns(3)
                    
                    with summary_col1:
                        st.metric("Models Used", result["model_count"])
                    with summary_col2:
                        st.metric("Successful Predictions", result["successful_predictions"])
                    with summary_col3:
                        consensus_risk = ensemble["consensus_risk"]
                        st.metric("Consensus Risk", consensus_risk)
                
                else:
                    st.error(f"Comparison failed: {response.text}")
        
        except requests.exceptions.RequestException:
            st.error("❌ Could not connect to the backend API. Please make sure the backend server is running on port 8000.")
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>💓 Heart Failure Prediction System v2.0 | Multi-Model Machine Learning Platform</p>
    <p>⚠️ This tool is for educational purposes only and should not replace professional medical advice.</p>
</div>
""", unsafe_allow_html=True)

# Installation instructions in sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📦 Installation")
    st.markdown("""
    **For full features with charts:**
    ```bash
    pip install plotly
    ```
    Then use `multi_model_app.py`
    
    **Current version:** Simple charts only
    """)
