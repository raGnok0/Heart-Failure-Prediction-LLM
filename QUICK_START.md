# Quick Start Guide - Heart Failure Prediction Multi-Model System

## 🚀 Get Running in 5 Minutes

### Option 1: Simple Setup (Recommended for Quick Testing)

1. **Install minimal dependencies:**
```bash
pip install streamlit requests pandas
```

2. **Start the backend:**
```bash
cd backend
python main.py
```
You should see: `Uvicorn running on http://0.0.0.0:8000`

3. **Start the frontend (in a new terminal):**
```bash
cd frontend
streamlit run multi_model_app_simple.py
```
You should see: `Local URL: http://localhost:8501`

4. **Open your browser and go to:** `http://localhost:8501`

### Option 2: Full Setup (With Advanced Charts)

1. **Install all dependencies:**
```bash
pip install -r requirements.txt
```

2. **Start the backend:**
```bash
cd backend
python main.py
```

3. **Start the enhanced frontend:**
```bash
cd frontend
streamlit run multi_model_app.py
```

## 🧪 Test the System

### Quick API Test
```bash
python test_system.py
```

### Manual Test
1. Open the frontend in your browser
2. Enter sample patient data:
   - Age: 75
   - Ejection Fraction: 20
   - Serum Creatinine: 1.9
   - (Fill in other fields)
3. Try different prediction modes:
   - Single Model
   - Compare All Models
   - Model Performance

## 🎯 What You'll See

### Single Model Mode
- Individual model predictions
- Risk level assessment
- Confidence scores

### Compare All Models Mode
- Side-by-side model comparison
- Ensemble prediction
- Consensus analysis
- Risk probability charts

### Model Performance Mode
- Accuracy comparison table
- Performance metrics
- Feature importance analysis

## 🔧 Troubleshooting

### Backend Issues
- **Port 8000 in use:** Change port in `backend/main.py`
- **Module not found:** Install missing dependencies
- **Model files missing:** Models are created automatically

### Frontend Issues
- **Plotly error:** Use `multi_model_app_simple.py` instead
- **Connection error:** Ensure backend is running on port 8000
- **Streamlit issues:** Try `pip install --upgrade streamlit`

### Common Fixes
```bash
# If you get import errors:
pip install fastapi uvicorn scikit-learn xgboost pandas numpy joblib

# If frontend won't start:
pip install streamlit requests pandas

# If you want full features:
pip install plotly
```

## 📊 Sample Test Data

Use this data for testing:

**High Risk Patient:**
- Age: 75, Ejection Fraction: 20, Serum Creatinine: 1.9
- Expected: High Risk prediction

**Low Risk Patient:**
- Age: 45, Ejection Fraction: 60, Serum Creatinine: 1.0
- Expected: Low Risk prediction

## 🎉 Success Indicators

✅ **Backend Working:** API docs at `http://localhost:8000/docs`
✅ **Frontend Working:** Interface at `http://localhost:8501`
✅ **Models Loaded:** Sidebar shows "✅ X models loaded"
✅ **Predictions Working:** Get results when clicking predict button

## 📞 Need Help?

1. **Check the logs** in your terminal for error messages
2. **Run the test script:** `python test_system.py`
3. **Try the simple version** if you have dependency issues
4. **Ensure Python 3.8+** is installed

## 🔄 Next Steps

Once running:
1. Explore different prediction modes
2. Compare model performance
3. Test with various patient data
4. Check feature importance analysis
5. Review the comprehensive documentation in README.md

---

**Time to get running:** ~5 minutes
**Dependencies:** Minimal (streamlit, requests, pandas)
**Full features:** Add plotly for enhanced charts
