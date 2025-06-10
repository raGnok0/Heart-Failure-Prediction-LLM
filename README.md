# Heart Failure Prediction - Multi-Model System

A comprehensive heart failure prediction system using multiple machine learning models with comparison capabilities and performance analysis.

## 🚀 Features

### Multi-Model Architecture
- **XGBoost**: Gradient boosting classifier (primary model)
- **Random Forest**: Ensemble learning method
- **Logistic Regression**: Linear classification model
- **Support Vector Machine (SVM)**: Kernel-based classifier

### Prediction Capabilities
- **Single Model Prediction**: Get predictions from individual models
- **Multi-Model Comparison**: Compare predictions across all models
- **Ensemble Prediction**: Combined prediction with consensus analysis
- **Performance Metrics**: Detailed accuracy and performance comparisons

### Advanced Features
- **Feature Importance Analysis**: Understand which factors matter most
- **Risk Level Assessment**: High/Medium/Low risk categorization
- **Confidence Scoring**: Model confidence in predictions
- **Interactive Dashboard**: User-friendly web interface

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| XGBoost | 0.8333 | 0.7500 | 0.7500 | 0.7500 | 0.8750 |
| Random Forest | 0.8167 | 0.7273 | 0.8000 | 0.7619 | 0.8625 |
| Logistic Regression | 0.8000 | 0.7000 | 0.7000 | 0.7000 | 0.8500 |
| SVM | 0.7833 | 0.6923 | 0.7500 | 0.7200 | 0.8375 |

**Best Overall Model**: XGBoost (based on F1-Score)

## 🏗️ Project Structure

```
Heart Failure Prediction/
├── backend/
│   ├── main.py                     # FastAPI server with multi-model endpoints
│   ├── model_comparison.py         # Model comparison and ensemble logic
│   ├── create_sample_models.py     # Script to create sample models
│   └── models/
│       ├── train_model.py          # Original single model training
│       ├── train_multiple_models.py # Multi-model training script
│       ├── heart_failure_model.pkl # Legacy XGBoost model
│       ├── xgboost_model.pkl       # XGBoost model
│       ├── random_forest_model.pkl # Random Forest model
│       ├── logistic_regression_model.pkl # Logistic Regression model
│       ├── scaler.pkl              # Legacy scaler
│       ├── standard_scaler.pkl     # Standard scaler for all models
│       ├── model_performance.json  # Performance metrics
│       └── feature_names.json      # Feature names
├── frontend/
│   ├── app.py                      # Original Streamlit app
│   └── multi_model_app.py          # Enhanced multi-model interface
├── data/
│   └── heart_failure_clinical_records_dataset.csv
└── README.md
```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Backend Setup
1. Install required packages:
```bash
pip install fastapi uvicorn scikit-learn xgboost pandas numpy joblib
```

2. Start the backend server:
```bash
cd backend
python main.py
```
The API will be available at `http://localhost:8000`

### Frontend Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the multi-model interface:
```bash
cd frontend
# For full features with interactive charts:
streamlit run multi_model_app.py

# Or for simple version (no plotly required):
streamlit run multi_model_app_simple.py
```

3. Or run the original interface:
```bash
streamlit run app.py
```

### Quick Start (No Dependencies)
If you encounter dependency issues, use the simple version:
```bash
pip install streamlit requests pandas
cd frontend
streamlit run multi_model_app_simple.py
```

## 📡 API Endpoints

### Prediction Endpoints
- `POST /predict` - Legacy single model prediction
- `POST /predict/single/{model_name}` - Predict with specific model
- `POST /predict/compare` - Compare predictions from all models

### Model Information Endpoints
- `GET /models/available` - List available models
- `GET /models/performance` - Get all model performance metrics
- `GET /models/performance/{model_name}` - Get specific model performance
- `GET /models/feature-importance` - Get feature importance for all models
- `GET /models/feature-importance/{model_name}` - Get feature importance for specific model
- `GET /models/info` - Get general model information

### Example API Usage

#### Single Model Prediction
```bash
curl -X POST "http://localhost:8000/predict/single/xgboost" \
     -H "Content-Type: application/json" \
     -d '{
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
     }'
```

#### Multi-Model Comparison
```bash
curl -X POST "http://localhost:8000/predict/compare" \
     -H "Content-Type: application/json" \
     -d '{...same input data...}'
```

## 📊 Input Features

The system requires 12 clinical features:

| Feature | Description | Type | Range |
|---------|-------------|------|-------|
| age | Patient age | Integer | 1-120 |
| anaemia | Decrease of red blood cells | Binary | 0/1 |
| creatinine_phosphokinase | Level of CPK enzyme | Integer | 0-10000 |
| diabetes | Diabetes status | Binary | 0/1 |
| ejection_fraction | Percentage of blood leaving heart | Integer | 0-100 |
| high_blood_pressure | Hypertension status | Binary | 0/1 |
| platelets | Platelet count | Float | 0-1000000 |
| serum_creatinine | Serum creatinine level | Float | 0.1-10.0 |
| serum_sodium | Serum sodium level | Integer | 100-150 |
| sex | Gender | Binary | 0=Female, 1=Male |
| smoking | Smoking status | Binary | 0/1 |
| time | Follow-up period (days) | Integer | 0-300 |

## 🎯 Model Comparison Features

### Individual Model Analysis
- Compare accuracy, precision, recall, F1-score, and ROC-AUC
- Feature importance rankings for each model
- Model-specific confidence scores

### Ensemble Prediction
- Average probability across all models
- Consensus risk level determination
- Consensus strength measurement

### Performance Dashboard
- Interactive charts and visualizations
- Model ranking by different metrics
- Feature importance comparison across models

## 🔬 Training New Models

To train new models with your own data:

1. **Prepare your dataset** in the same format as the provided CSV
2. **Run the training script**:
```bash
cd backend/models
python train_multiple_models.py
```

3. **Or use the comprehensive trainer**:
```bash
python train_multiple_models.py
```

This will:
- Train all 4 models
- Generate performance metrics
- Save models and scalers
- Create comparison visualizations

## 🚨 Important Notes

### Medical Disclaimer
⚠️ **This system is for educational and research purposes only. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment.**

### Model Limitations
- Models are trained on a specific dataset and may not generalize to all populations
- Performance metrics are based on historical data
- Real-world performance may vary

### Data Privacy
- No patient data is stored by the system
- All predictions are processed in real-time
- Input data is not logged or saved

## 🔄 System Workflow

1. **Data Input**: User enters patient clinical data
2. **Model Selection**: Choose single model or comparison mode
3. **Prediction**: System processes data through selected models
4. **Risk Assessment**: Calculate risk levels and confidence scores
5. **Results Display**: Present predictions with visualizations
6. **Performance Analysis**: Compare model performance and feature importance

## 🛠️ Customization

### Adding New Models
1. Train your model using the same feature set
2. Save the model as a `.pkl` file in the `backend/models/` directory
3. Update the `model_comparison.py` file to include your model
4. Add performance metrics to `model_performance.json`

### Modifying Risk Thresholds
Risk levels are determined by probability thresholds:
- **High Risk**: ≥ 70%
- **Medium Risk**: 40-69%
- **Low Risk**: < 40%

These can be modified in the `model_comparison.py` file.

## 📈 Future Enhancements

- [ ] Deep learning models (Neural Networks)
- [ ] Real-time model retraining
- [ ] Advanced ensemble methods
- [ ] Model explainability (SHAP values)
- [ ] A/B testing framework
- [ ] Model drift detection
- [ ] Integration with medical databases

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is for educational purposes. Please ensure compliance with medical data regulations in your jurisdiction.

## 📞 Support

For questions or issues:
1. Check the API documentation at `http://localhost:8000/docs`
2. Review the model performance metrics
3. Ensure all dependencies are installed correctly

---

**Version**: 2.0.0  
**Last Updated**: December 2024  
**Models**: XGBoost, Random Forest, Logistic Regression, SVM
