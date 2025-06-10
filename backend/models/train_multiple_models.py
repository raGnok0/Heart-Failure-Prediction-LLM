import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class MultiModelTrainer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.models = {}
        self.scalers = {}
        self.performance_metrics = {}
        self.feature_names = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess the heart failure dataset"""
        print("Loading and preprocessing data...")
        
        # Load the data
        df = pd.read_csv(self.data_path)
        
        # Separate features and target
        X = df.drop(columns=['DEATH_EVENT'])
        y = df['DEATH_EVENT']
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features for models that need it
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['standard'] = scaler
        
        return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
    
    def initialize_models(self):
        """Initialize all models with optimized parameters"""
        self.models = {
            'xgboost': XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=1000
            ),
            'svm': SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )
        }
    
    def train_models(self, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled):
        """Train all models and evaluate their performance"""
        print("Training models...")
        
        model_data = {
            'xgboost': (X_train, X_test),
            'random_forest': (X_train, X_test),
            'logistic_regression': (X_train_scaled, X_test_scaled),
            'svm': (X_train_scaled, X_test_scaled)
        }
        
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")
            
            # Get appropriate data (scaled or unscaled)
            X_train_model, X_test_model = model_data[model_name]
            
            # Train the model
            model.fit(X_train_model, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_model)
            y_pred_proba = model.predict_proba(X_test_model)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'cross_val_accuracy': np.mean(cross_val_score(model, X_train_model, y_train, cv=5))
            }
            
            # Store performance metrics
            self.performance_metrics[model_name] = metrics
            
            # Get feature importance if available
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(self.feature_names, model.feature_importances_))
                self.performance_metrics[model_name]['feature_importance'] = feature_importance
            elif hasattr(model, 'coef_'):
                feature_importance = dict(zip(self.feature_names, abs(model.coef_[0])))
                self.performance_metrics[model_name]['feature_importance'] = feature_importance
            
            print(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
    
    def save_models_and_metrics(self):
        """Save all trained models and performance metrics"""
        print("\nSaving models and metrics...")
        
        # Save models
        for model_name, model in self.models.items():
            joblib.dump(model, f'backend/models/{model_name}_model.pkl')
        
        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            joblib.dump(scaler, f'backend/models/{scaler_name}_scaler.pkl')
        
        # Save performance metrics
        with open('backend/models/model_performance.json', 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            metrics_json = {}
            for model_name, metrics in self.performance_metrics.items():
                metrics_json[model_name] = {}
                for key, value in metrics.items():
                    if isinstance(value, dict):
                        metrics_json[model_name][key] = {k: float(v) for k, v in value.items()}
                    else:
                        metrics_json[model_name][key] = float(value)
            
            json.dump(metrics_json, f, indent=4)
        
        # Save feature names
        with open('backend/models/feature_names.json', 'w') as f:
            json.dump(self.feature_names, f)
    
    def generate_comparison_report(self):
        """Generate a detailed comparison report"""
        print("\nGenerating comparison report...")
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(self.performance_metrics).T
        comparison_df = comparison_df.drop('feature_importance', axis=1, errors='ignore')
        
        print("\n" + "="*60)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*60)
        print(comparison_df.round(4))
        
        # Find best model for each metric
        print("\n" + "="*60)
        print("BEST MODELS BY METRIC")
        print("="*60)
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
            best_model = comparison_df[metric].idxmax()
            best_score = comparison_df[metric].max()
            print(f"{metric.upper()}: {best_model} ({best_score:.4f})")
        
        # Overall best model (based on F1 score)
        overall_best = comparison_df['f1_score'].idxmax()
        print(f"\nOVERALL BEST MODEL (F1-Score): {overall_best}")
        
        return comparison_df
    
    def create_visualizations(self):
        """Create performance visualization plots"""
        print("Creating performance visualizations...")
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(self.performance_metrics).T
        comparison_df = comparison_df.drop('feature_importance', axis=1, errors='ignore')
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Accuracy Comparison
        axes[0, 0].bar(comparison_df.index, comparison_df['accuracy'], color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: ROC-AUC Comparison
        axes[0, 1].bar(comparison_df.index, comparison_df['roc_auc'], color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('ROC-AUC Comparison')
        axes[0, 1].set_ylabel('ROC-AUC')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: F1-Score Comparison
        axes[1, 0].bar(comparison_df.index, comparison_df['f1_score'], color='lightcoral', alpha=0.7)
        axes[1, 0].set_title('F1-Score Comparison')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: All Metrics Heatmap
        metrics_for_heatmap = comparison_df[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']]
        sns.heatmap(metrics_for_heatmap.T, annot=True, cmap='YlOrRd', ax=axes[1, 1], fmt='.3f')
        axes[1, 1].set_title('All Metrics Heatmap')
        
        plt.tight_layout()
        plt.savefig('backend/models/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Visualizations saved as 'model_comparison.png'")

def main():
    """Main function to run the multi-model training pipeline"""
    print("Starting Multi-Model Heart Failure Prediction Training...")
    print("="*60)
    
    # Initialize trainer
    trainer = MultiModelTrainer('data/heart_failure_clinical_records_dataset.csv')
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = trainer.load_and_preprocess_data()
    
    # Initialize models
    trainer.initialize_models()
    
    # Train models
    trainer.train_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)
    
    # Save models and metrics
    trainer.save_models_and_metrics()
    
    # Generate comparison report
    comparison_df = trainer.generate_comparison_report()
    
    # Create visualizations
    trainer.create_visualizations()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Models saved in 'backend/models/' directory")
    print("Performance metrics saved as 'model_performance.json'")
    print("Comparison visualization saved as 'model_comparison.png'")

if __name__ == "__main__":
    main()
