import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import joblib
import logging
from typing import Dict, Any, List
import shap
from pathlib import Path

from .utils import load_config, save_model_metrics

class ModelTrainer:
    """Train and evaluate machine learning models for song popularity prediction."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize ModelTrainer with configuration."""
        self.config = load_config(config_path)
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.feature_importance = {}
        self.shap_values = {}
        
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train Random Forest model."""
        try:
            rf_config = self.config['models']['random_forest']
            rf_model = RandomForestRegressor(**rf_config)
            rf_model.fit(X_train, y_train)
            self.models['random_forest'] = rf_model
            self.logger.info("Random Forest model trained successfully")
            
        except Exception as e:
            self.logger.error(f"Error training Random Forest model: {str(e)}")
            raise
            
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train XGBoost model."""
        try:
            xgb_config = self.config['models']['xgboost']
            xgb_model = xgb.XGBRegressor(**xgb_config)
            xgb_model.fit(X_train, y_train)
            self.models['xgboost'] = xgb_model
            self.logger.info("XGBoost model trained successfully")
            
        except Exception as e:
            self.logger.error(f"Error training XGBoost model: {str(e)}")
            raise
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        """Evaluate all trained models and return performance metrics."""
        results = {}
        try:
            for name, model in self.models.items():
                y_pred = model.predict(X_test)
                metrics = {
                    'R2 Score': r2_score(y_test, y_pred),
                    'MSE': mean_squared_error(y_test, y_pred),
                    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'MAE': mean_absolute_error(y_test, y_pred)
                }
                results[name] = metrics
                save_model_metrics(metrics, name, 'models/metrics')
                self.logger.info(f"Evaluation metrics calculated for {name}")
                
            return results
            
        except Exception as e:
            self.logger.error(f"Error evaluating models: {str(e)}")
            raise
    
    def calculate_feature_importance(self, X_train: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Calculate feature importance for all models using SHAP values."""
        try:
            importance_dict = {}
            for name, model in self.models.items():
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_train)
                
                # Store SHAP values for later use
                self.shap_values[name] = {
                    'values': shap_values,
                    'explainer': explainer
                }
                
                # Calculate mean absolute SHAP values for feature importance
                importance = pd.DataFrame({
                    'Feature': X_train.columns,
                    'Importance': np.abs(shap_values).mean(axis=0)
                })
                importance = importance.sort_values('Importance', ascending=False)
                importance_dict[name] = importance
                
            self.feature_importance = importance_dict
            self.logger.info("Feature importance calculated successfully")
            return importance_dict
            
        except Exception as e:
            self.logger.error(f"Error calculating feature importance: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Make predictions using all trained models."""
        predictions = {}
        try:
            for name, model in self.models.items():
                predictions[name] = model.predict(X)
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def save_models(self, path: str) -> None:
        """Save trained models to disk."""
        try:
            save_path = Path(path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            for name, model in self.models.items():
                model_path = save_path / f"{name}_model.joblib"
                joblib.dump(model, model_path)
            self.logger.info(f"Models saved successfully to {path}")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")
            raise
    
    def load_models(self, path: str) -> None:
        """Load trained models from disk."""
        try:
            load_path = Path(path)
            self.models['random_forest'] = joblib.load(load_path / "random_forest_model.joblib")
            self.models['xgboost'] = joblib.load(load_path / "xgboost_model.joblib")
            self.logger.info(f"Models loaded successfully from {path}")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            raise
    
    def get_feature_recommendations(self, input_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate recommendations for improving song popularity."""
        try:
            base_predictions = self.predict(input_data)
            base_popularity = np.mean([pred[0] for pred in base_predictions.values()])
            
            recommendations = []
            features_to_test = ['Energy', 'Danceability', 'Loudness', 'Valence']
            
            for feature in features_to_test:
                if feature in input_data.columns:
                    # Test increasing the feature
                    test_data_inc = input_data.copy()
                    test_data_inc[feature] *= 1.1
                    pred_inc = np.mean([pred[0] for pred in self.predict(test_data_inc).values()])
                    
                    # Test decreasing the feature
                    test_data_dec = input_data.copy()
                    test_data_dec[feature] *= 0.9
                    pred_dec = np.mean([pred[0] for pred in self.predict(test_data_dec).values()])
                    
                    # Calculate impact and direction
                    impact = max(abs(pred_inc - base_popularity), abs(pred_dec - base_popularity))
                    direction = "increase" if pred_inc > pred_dec else "decrease"
                    
                    recommendations.append({
                        'feature': feature,
                        'impact': impact,
                        'direction': direction,
                        'potential_improvement': max(pred_inc, pred_dec) - base_popularity
                    })
            
            # Sort recommendations by impact
            recommendations.sort(key=lambda x: abs(x['potential_improvement']), reverse=True)
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            raise