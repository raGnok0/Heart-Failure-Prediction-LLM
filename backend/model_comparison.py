import joblib
import numpy as np
import json
import os
from typing import Dict, List, Any, Tuple
from sklearn.preprocessing import StandardScaler

class ModelComparison:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.performance_metrics = {}
        self.feature_names = []
        self.load_models_and_data()
    
    def load_models_and_data(self):
        """Load all trained models, scalers, and performance data"""
        models_dir = 'models'
        
        # Load models
        model_files = {
            'xgboost': 'xgboost_model.pkl',
            'random_forest': 'random_forest_model.pkl',
            'logistic_regression': 'logistic_regression_model.pkl',
            'svm': 'svm_model.pkl'
        }
        
        for model_name, filename in model_files.items():
            model_path = os.path.join(models_dir, filename)
            if os.path.exists(model_path):
                self.models[model_name] = joblib.load(model_path)
        
        # Load scalers
        scaler_files = {
            'standard': 'standard_scaler.pkl',
            'legacy': 'scaler.pkl'  # For backward compatibility
        }
        
        for scaler_name, filename in scaler_files.items():
            scaler_path = os.path.join(models_dir, filename)
            if os.path.exists(scaler_path):
                self.scalers[scaler_name] = joblib.load(scaler_path)
        
        # Load performance metrics
        metrics_path = os.path.join(models_dir, 'model_performance.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                self.performance_metrics = json.load(f)
        
        # Load feature names
        features_path = os.path.join(models_dir, 'feature_names.json')
        if os.path.exists(features_path):
            with open(features_path, 'r') as f:
                self.feature_names = json.load(f)
        else:
            # Default feature names if file doesn't exist
            self.feature_names = [
                'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
                'ejection_fraction', 'high_blood_pressure', 'platelets',
                'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time'
            ]
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return list(self.models.keys())
    
    def predict_single_model(self, model_name: str, features: np.ndarray) -> Dict[str, Any]:
        """Make prediction using a single model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Determine if model needs scaled features
        needs_scaling = model_name in ['logistic_regression', 'svm']
        
        if needs_scaling:
            scaler = self.scalers.get('standard', self.scalers.get('legacy'))
            if scaler:
                features = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        # Calculate risk level
        risk_probability = probability[1] * 100
        if risk_probability >= 70:
            risk_level = "High Risk"
        elif risk_probability >= 40:
            risk_level = "Medium Risk"
        else:
            risk_level = "Low Risk"
        
        return {
            'model_name': model_name,
            'prediction': "Heart Failure Risk Detected" if prediction == 1 else "No Heart Failure Risk",
            'risk_level': risk_level,
            'probability': f"{risk_probability:.2f}%",
            'confidence': f"{max(probability) * 100:.2f}%"
        }
    
    def predict_all_models(self, features: np.ndarray) -> Dict[str, Any]:
        """Make predictions using all available models"""
        predictions = {}
        probabilities = []
        risk_votes = {'High Risk': 0, 'Medium Risk': 0, 'Low Risk': 0}
        
        for model_name in self.models.keys():
            try:
                result = self.predict_single_model(model_name, features)
                predictions[model_name] = result
                
                # Extract probability for ensemble
                prob = float(result['probability'].replace('%', ''))
                probabilities.append(prob)
                
                # Count risk level votes
                risk_votes[result['risk_level']] += 1
                
            except Exception as e:
                predictions[model_name] = {
                    'error': str(e),
                    'prediction': 'Error',
                    'risk_level': 'Unknown',
                    'probability': '0.00%',
                    'confidence': '0.00%'
                }
        
        # Calculate ensemble prediction
        if probabilities:
            avg_probability = np.mean(probabilities)
            ensemble_prediction = "Heart Failure Risk Detected" if avg_probability >= 50 else "No Heart Failure Risk"
            
            # Determine ensemble risk level
            if avg_probability >= 70:
                ensemble_risk = "High Risk"
            elif avg_probability >= 40:
                ensemble_risk = "Medium Risk"
            else:
                ensemble_risk = "Low Risk"
            
            # Find consensus risk level
            consensus_risk = max(risk_votes, key=risk_votes.get)
            consensus_strength = risk_votes[consensus_risk] / len(self.models)
            
        else:
            avg_probability = 0
            ensemble_prediction = "Error in prediction"
            ensemble_risk = "Unknown"
            consensus_risk = "Unknown"
            consensus_strength = 0
        
        return {
            'individual_predictions': predictions,
            'ensemble_prediction': {
                'prediction': ensemble_prediction,
                'risk_level': ensemble_risk,
                'average_probability': f"{avg_probability:.2f}%",
                'consensus_risk': consensus_risk,
                'consensus_strength': f"{consensus_strength * 100:.1f}%"
            },
            'model_count': len(self.models),
            'successful_predictions': len([p for p in predictions.values() if 'error' not in p])
        }
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all models"""
        if not self.performance_metrics:
            return {"error": "Performance metrics not available"}
        
        # Calculate rankings
        rankings = {}
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        for metric in metrics:
            metric_scores = {model: data.get(metric, 0) for model, data in self.performance_metrics.items()}
            sorted_models = sorted(metric_scores.items(), key=lambda x: x[1], reverse=True)
            rankings[metric] = [{'model': model, 'score': score} for model, score in sorted_models]
        
        # Find overall best model (based on F1 score)
        f1_scores = {model: data.get('f1_score', 0) for model, data in self.performance_metrics.items()}
        best_model = max(f1_scores, key=f1_scores.get) if f1_scores else None
        
        return {
            'performance_metrics': self.performance_metrics,
            'rankings': rankings,
            'best_overall_model': best_model,
            'available_models': list(self.performance_metrics.keys())
        }
    
    def get_feature_importance(self, model_name: str = None) -> Dict[str, Any]:
        """Get feature importance for a specific model or all models"""
        if model_name:
            if model_name not in self.performance_metrics:
                return {"error": f"Model {model_name} not found"}
            
            feature_importance = self.performance_metrics[model_name].get('feature_importance', {})
            if not feature_importance:
                return {"error": f"Feature importance not available for {model_name}"}
            
            # Sort by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            return {
                'model': model_name,
                'feature_importance': dict(sorted_features),
                'top_features': sorted_features[:5]
            }
        else:
            # Get feature importance for all models
            all_importance = {}
            for model in self.performance_metrics.keys():
                feature_importance = self.performance_metrics[model].get('feature_importance', {})
                if feature_importance:
                    all_importance[model] = feature_importance
            
            return {
                'all_models': all_importance,
                'available_models': list(all_importance.keys())
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get general information about all models"""
        return {
            'available_models': self.get_available_models(),
            'feature_names': self.feature_names,
            'total_features': len(self.feature_names),
            'models_loaded': len(self.models),
            'scalers_available': list(self.scalers.keys()),
            'performance_data_available': bool(self.performance_metrics)
        }

# Global instance
model_comparison = ModelComparison()
