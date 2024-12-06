import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import json
from pathlib import Path
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
from src.utils import load_config, setup_logging

def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        'models/saved_models',
        'models/metrics',
        'logs'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def save_training_metadata(metrics, model_params, features, output_path):
    """Save training metadata and metrics."""
    metadata = {
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_metrics': metrics,
        'model_parameters': model_params,
        'features_used': features
    }
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / 'training_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)

def evaluate_models(trainer, X_test, y_test):
    """Evaluate models and print results."""
    results = trainer.evaluate_models(X_test, y_test)
    
    print("\nModel Performance Metrics:")
    print("-" * 50)
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()} METRICS:")
        print(f"R² Score: {metrics['R2 Score']:.4f}")
        print(f"RMSE: {metrics['RMSE']:.4f}")
        print(f"MAE: {metrics['MAE']:.4f}")
    
    return results

def analyze_feature_importance(trainer, X_train, output_path):
    """Analyze and save feature importance."""
    importance_dict = trainer.calculate_feature_importance(X_train)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for model_name, importance_df in importance_dict.items():
        importance_df.to_csv(
            output_path / f'{model_name}_feature_importance.csv',
            index=False
        )
        
        print(f"\n{model_name.upper()} - Top 10 Important Features:")
        print(importance_df.head(10))

def main():
    """Main training script."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        print("\n=== Starting Model Training Process ===\n")
        
        # Create necessary directories
        setup_directories()
        print("✓ Directories created successfully")
        
        # Load configuration
        config = load_config()
        data_path = "data/Spotify_data.csv"  # Direct path specification
        print(f"✓ Configuration loaded, using data from: {data_path}")
        
        # Check if data file exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")
        
        # Initialize components
        processor = DataProcessor()
        trainer = ModelTrainer()
        
        # Load and process data
        print("\nStep 1: Loading and Processing Data...")
        data = processor.load_data(data_path)
        print(f"✓ Data loaded successfully. Shape: {data.shape}")
        
        # Preprocess data
        X, y = processor.preprocess_data()
        X_train, X_test, y_train, y_test = processor.get_train_test_split(X, y)
        print(f"✓ Data preprocessing complete. Training set shape: {X_train.shape}")
        
        # Train models
        print("\nStep 2: Training Models...")
        print("Training Random Forest model...")
        trainer.train_random_forest(X_train, y_train)
        print("✓ Random Forest training complete")
        
        print("\nTraining XGBoost model...")
        trainer.train_xgboost(X_train, y_train)
        print("✓ XGBoost training complete")
        
        # Evaluate models
        print("\nStep 3: Evaluating Models...")
        results = evaluate_models(trainer, X_test, y_test)
        
        # Analyze feature importance
        print("\nStep 4: Analyzing Feature Importance...")
        importance_dict = trainer.calculate_feature_importance(X_train)
        
        # Save models and metadata
        print("\nStep 5: Saving Models and Metadata...")
        models_dir = Path('models/saved_models')
        metrics_dir = Path('models/metrics')
        
        # Save models
        trainer.save_models(str(models_dir))
        
        # Save scaler
        joblib.dump(processor.scaler, models_dir / 'scaler.joblib')
        
        # Save metadata
        model_params = {
            'random_forest': trainer.models['random_forest'].get_params(),
            'xgboost': trainer.models['xgboost'].get_params()
        }
        
        save_training_metadata(
            results,
            model_params,
            X_train.columns.tolist(),
            metrics_dir
        )
        
        # Save feature importance
        analyze_feature_importance(trainer, X_train, metrics_dir)
        
        print("\n=== Training Process Completed Successfully! ===")
        print("\nSaved files:")
        print(f"Models directory: {models_dir}")
        print(f"Metrics directory: {metrics_dir}")
        print("\nYou can now run the Streamlit app using:")
        print("streamlit run app/app.py")
        
    except Exception as e:
        logger.error(f"Error in training process: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()