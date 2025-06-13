# Heart Failure Prediction System - Documentation (Part 2)

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Backend Issues

**Issue: "ModuleNotFoundError: No module named 'xgboost'"**
```bash
# Solution: Install missing dependencies
pip install xgboost
# or install all requirements
pip install -r requirements.txt
```

**Issue: "FileNotFoundError: Model file not found"**
```bash
# Solution: Generate model files
cd backend
python create_sample_models.py
```

**Issue: "Port 8000 already in use"**
```bash
# Solution 1: Kill existing process
lsof -ti:8000 | xargs kill -9

# Solution 2: Use different port
# Edit backend/main.py and change port
uvicorn.run(app, host="0.0.0.0", port=8001)
```

**Issue: "CORS policy error"**
```python
# Solution: Update CORS settings in main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

#### Frontend Issues

**Issue: "ModuleNotFoundError: No module named 'plotly'"**
```bash
# Solution 1: Install plotly
pip install plotly

# Solution 2: Use simple version
streamlit run frontend/multi_model_app_simple.py
```

**Issue: "Connection refused to backend"**
```python
# Solution: Check backend URL in frontend code
# Update API base URL if needed
BASE_URL = "http://127.0.0.1:8000"  # or your backend URL
```

**Issue: "Streamlit app won't start"**
```bash
# Solution: Clear Streamlit cache
streamlit cache clear

# Check Streamlit version
pip install --upgrade streamlit
```

#### Model Issues

**Issue: "Model prediction returns NaN"**
```python
# Solution: Check input data validation
def validate_input_data(data):
    required_fields = [
        'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
        'ejection_fraction', 'high_blood_pressure', 'platelets',
        'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time'
    ]
    
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
        if pd.isna(data[field]):
            raise ValueError(f"Field {field} cannot be NaN")
    
    return True
```

**Issue: "Inconsistent model predictions"**
```python
# Solution: Check model versions and data preprocessing
def check_model_consistency():
    # Load test data
    test_features = np.array([[65, 0, 582, 1, 40, 1, 265000, 1.9, 130, 1, 0, 4]])
    
    # Test all models
    for model_name in ['xgboost', 'random_forest', 'logistic_regression', 'svm']:
        try:
            result = model_comparison.predict_single_model(model_name, test_features)
            print(f"{model_name}: {result['probability']}")
        except Exception as e:
            print(f"Error with {model_name}: {e}")
```

### Performance Issues

**Issue: "Slow API response times"**
```python
# Solution 1: Implement caching
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_prediction(model_name, features_hash):
    return model_comparison.predict_single_model(model_name, features)

# Solution 2: Optimize model loading
class LazyModelLoader:
    def __init__(self):
        self._models = {}
    
    def get_model(self, model_name):
        if model_name not in self._models:
            self._models[model_name] = joblib.load(f'models/{model_name}_model.pkl')
        return self._models[model_name]
```

**Issue: "High memory usage"**
```python
# Solution: Implement memory monitoring
import psutil
import gc

def monitor_memory():
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
    
    # Force garbage collection if memory usage is high
    if memory_info.rss > 500 * 1024 * 1024:  # 500MB threshold
        gc.collect()
```

### Debugging Tools

#### API Debugging

```python
# backend/debug.py
import logging
from fastapi import Request
import time

@app.middleware("http")
async def debug_middleware(request: Request, call_next):
    start_time = time.time()
    
    # Log request
    logging.info(f"Request: {request.method} {request.url}")
    
    response = await call_next(request)
    
    # Log response time
    process_time = time.time() - start_time
    logging.info(f"Response time: {process_time:.4f}s")
    
    return response
```

#### Model Debugging

```python
# backend/model_debug.py
def debug_model_prediction(model_name, features):
    """Debug model prediction step by step"""
    print(f"Debugging {model_name} prediction...")
    
    # Check input features
    print(f"Input features shape: {features.shape}")
    print(f"Input features: {features}")
    
    # Load model
    model = model_comparison.get_model(model_name)
    print(f"Model type: {type(model)}")
    
    # Check if scaling is needed
    if model_name in ['logistic_regression', 'svm']:
        scaler = model_comparison.get_scaler('standard')
        scaled_features = scaler.transform(features)
        print(f"Scaled features: {scaled_features}")
        features = scaled_features
    
    # Make prediction
    prediction = model.predict(features)
    probability = model.predict_proba(features)
    
    print(f"Raw prediction: {prediction}")
    print(f"Raw probability: {probability}")
    
    return prediction, probability
```

---

## 🤝 Contributing

### How to Contribute

We welcome contributions to the Heart Failure Prediction System! Here's how you can help:

#### Types of Contributions

1. **Bug Reports**: Report issues you encounter
2. **Feature Requests**: Suggest new features or improvements
3. **Code Contributions**: Submit bug fixes or new features
4. **Documentation**: Improve or expand documentation
5. **Testing**: Add or improve test coverage

#### Getting Started

1. **Fork the Repository**
   ```bash
   git clone https://github.com/your-username/heart-failure-prediction.git
   cd heart-failure-prediction
   ```

2. **Set Up Development Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

#### Development Guidelines

**Code Style:**
- Follow PEP 8 for Python code
- Use type hints for all functions
- Write docstrings for all public functions
- Maximum line length: 88 characters

**Testing:**
- Write tests for all new features
- Ensure all existing tests pass
- Aim for >90% test coverage

**Documentation:**
- Update documentation for any API changes
- Include examples for new features
- Update README if necessary

#### Submitting Changes

1. **Run Tests**
   ```bash
   pytest tests/
   flake8 backend/ frontend/
   black backend/ frontend/
   mypy backend/
   ```

2. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new model comparison feature"
   ```

3. **Push to Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Create Pull Request**
   - Go to GitHub and create a pull request
   - Provide clear description of changes
   - Link any related issues

#### Code Review Process

1. **Automated Checks**: CI/CD pipeline runs tests
2. **Peer Review**: Code reviewed by maintainers
3. **Feedback**: Address any requested changes
4. **Merge**: Approved changes are merged

### Development Roadmap

#### Short-term Goals (Next 3 months)

1. **Enhanced Security**
   - JWT authentication implementation
   - API rate limiting
   - Input validation improvements

2. **Performance Optimization**
   - Model caching implementation
   - Database integration for results storage
   - Async prediction processing

3. **Additional Models**
   - Neural network implementation
   - Ensemble method improvements
   - Model versioning system

#### Medium-term Goals (3-6 months)

1. **Advanced Features**
   - SHAP explainability integration
   - A/B testing framework
   - Real-time model monitoring

2. **Clinical Integration**
   - HL7 FHIR compatibility
   - EHR integration templates
   - Clinical decision support tools

3. **Deployment Improvements**
   - Kubernetes helm charts
   - CI/CD pipeline enhancements
   - Multi-environment support

#### Long-term Goals (6+ months)

1. **Research Features**
   - Federated learning support
   - Multi-center validation
   - Longitudinal outcome tracking

2. **Platform Expansion**
   - Mobile application
   - Cloud-native architecture
   - Multi-language support

### Issue Templates

#### Bug Report Template

```markdown
**Bug Description**
A clear description of the bug.

**Steps to Reproduce**
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Environment**
- OS: [e.g. Windows 10, macOS 11.0, Ubuntu 20.04]
- Python Version: [e.g. 3.9.0]
- Browser: [e.g. Chrome 95.0]

**Additional Context**
Any other context about the problem.
```

#### Feature Request Template

```markdown
**Feature Description**
A clear description of the feature you'd like to see.

**Use Case**
Describe the use case or problem this feature would solve.

**Proposed Solution**
Describe how you envision this feature working.

**Alternatives Considered**
Any alternative solutions you've considered.

**Additional Context**
Any other context or screenshots about the feature request.
```

---

## 📄 License & Disclaimer

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Heart Failure Prediction System

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Medical Disclaimer

**⚠️ IMPORTANT MEDICAL DISCLAIMER ⚠️**

This software is provided for **educational and research purposes only**. It is **NOT intended for clinical use** or as a substitute for professional medical advice, diagnosis, or treatment.

#### Key Points:

1. **Not a Medical Device**: This system has not been approved by the FDA or any other regulatory body as a medical device.

2. **Educational Purpose**: The predictions are based on machine learning models trained on historical data and should be used only for educational and research purposes.

3. **No Clinical Validation**: While the models show good performance on test data, they have not undergone clinical validation required for medical use.

4. **Professional Medical Advice**: Always seek the advice of qualified healthcare professionals for any medical concerns or decisions.

5. **No Liability**: The developers and contributors assume no responsibility for any consequences resulting from the use of this software.

6. **Data Privacy**: Ensure compliance with applicable healthcare data privacy regulations (HIPAA, GDPR, etc.) when using this system.

#### Recommended Use Cases:

✅ **Appropriate Uses:**
- Educational demonstrations
- Research and development
- Algorithm comparison studies
- Software development learning
- Academic coursework

❌ **Inappropriate Uses:**
- Clinical decision making
- Patient diagnosis
- Treatment planning
- Medical screening
- Any direct patient care

### Data Attribution

The heart failure dataset used in this project is from:

**Citation:**
Davide Chicco, Giuseppe Jurman: Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. BMC Medical Informatics and Decision Making 20, 16 (2020).

**Dataset Source:**
UCI Machine Learning Repository: Heart failure clinical records Data Set
https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records

### Acknowledgments

- **Dataset Contributors**: Tanvir Ahmad, Assia Munir, Sajjad Haider Bhatti, Muhammad Aftab, and Muhammad Ali Raza
- **Research Community**: All researchers working on heart failure prediction
- **Open Source Libraries**: Scikit-learn, XGBoost, FastAPI, Streamlit, and other dependencies
- **Contributors**: All developers who have contributed to this project

### Contact Information

For questions, suggestions, or collaboration opportunities:

- **Project Repository**: [GitHub Repository URL]
- **Issues**: [GitHub Issues URL]
- **Discussions**: [GitHub Discussions URL]
- **Email**: [Contact Email]

### Version History

- **v2.0.0** (Current): Multi-model comparison system with advanced frontend
- **v1.0.0**: Initial single-model implementation
- **v0.1.0**: Proof of concept

---

## 📚 Additional Resources

### Research Papers

1. **Chicco, D., & Jurman, G. (2020)**. Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. *BMC Medical Informatics and Decision Making*, 20(1), 16.

2. **Ahmad, T., et al. (2017)**. Clinical implications of chronic heart failure phenotypes defined by cluster analysis. *JACC: Heart Failure*, 2(3), 271-280.

3. **Ishaq, A., et al. (2021)**. Improving the prediction of heart failure patients' survival using SMOTE and effective data mining techniques. *IEEE Access*, 9, 39707-39716.

### Online Resources

- **Heart Failure Society of America**: https://www.hfsa.org/
- **American Heart Association**: https://www.heart.org/
- **European Society of Cardiology**: https://www.escardio.org/
- **Machine Learning in Healthcare**: https://www.mlinhealthcare.org/

### Technical Documentation

- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **Streamlit Documentation**: https://docs.streamlit.io/
- **Scikit-learn Documentation**: https://scikit-learn.org/
- **XGBoost Documentation**: https://xgboost.readthedocs.io/

### Datasets for Further Research

1. **MIMIC-III Clinical Database**: https://mimic.physionet.org/
2. **PhysioNet Databases**: https://physionet.org/
3. **Kaggle Heart Disease Datasets**: https://www.kaggle.com/datasets?search=heart+disease
4. **UCI ML Repository**: https://archive.ics.uci.edu/ml/

---

**End of Documentation**

*This comprehensive documentation covers all aspects of the Heart Failure Prediction System. For the most up-to-date information, please refer to the project repository.*
