import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import json
import os

def create_sample_models():
    """Create sample models for testing the multi-model system"""
    print("Creating sample models for testing...")
    
    # Load the data
    df = pd.read_csv('../data/heart_failure_clinical_records_dataset.csv')
    
    # Separate features and target
    X = df.drop(columns=['DEATH_EVENT'])
    y = df['DEATH_EVENT']
    
    # Store feature names
    feature_names = X.columns.tolist()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'xgboost': XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
        'svm': SVC(kernel='rbf', probability=True, random_state=42)
    }
    
    # Data for each model (scaled vs unscaled)
    model_data = {
        'xgboost': (X_train, X_test),
        'random_forest': (X_train, X_test),
        'logistic_regression': (X_train_scaled, X_test_scaled),
        'svm': (X_train_scaled, X_test_scaled)
    }
    
    performance_metrics = {}
    
    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        
        # Get appropriate data
        X_train_model, X_test_model = model_data[model_name]
        
        # Train the model
        model.fit(X_train_model, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_model)
        y_pred_proba = model.predict_proba(X_test_model)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'f1_score': float(f1_score(y_test, y_pred)),
            'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
            'cross_val_accuracy': 0.85  # Placeholder
        }
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(feature_names, [float(x) for x in model.feature_importances_]))
        elif hasattr(model, 'coef_'):
            feature_importance = dict(zip(feature_names, [float(x) for x in abs(model.coef_[0])]))
        else:
            feature_importance = {}
        
        metrics['feature_importance'] = feature_importance
        performance_metrics[model_name] = metrics
        
        # Save the model
        joblib.dump(model, f'{model_name}_model.pkl')
        print(f"Saved {model_name}_model.pkl")
    
    # Save the scaler
    joblib.dump(scaler, 'standard_scaler.pkl')
    print("Saved standard_scaler.pkl")
    
    # Save performance metrics
    with open('model_performance.json', 'w') as f:
        json.dump(performance_metrics, f, indent=4)
    print("Saved model_performance.json")
    
    # Save feature names
    with open('feature_names.json', 'w') as f:
        json.dump(feature_names, f)
    print("Saved feature_names.json")
    
    print("\nModel Performance Summary:")
    print("-" * 50)
    for model_name, metrics in performance_metrics.items():
        print(f"{model_name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        print()
    
    print("Sample models created successfully!")

if __name__ == "__main__":
    create_sample_models()
