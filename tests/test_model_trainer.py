import pytest
import pandas as pd
import numpy as np
from src.model_trainer import ModelTrainer
import os

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    X = pd.DataFrame({
        'Feature1': np.random.rand(100),
        'Feature2': np.random.rand(100),
        'Feature3': np.random.rand(100)
    })
    y = np.random.randint(0, 100, 100)
    return X, pd.Series(y)

@pytest.fixture
def trainer():
    """Create ModelTrainer instance."""
    return ModelTrainer()

def test_random_forest_training(trainer, sample_data):
    """Test Random Forest model training."""
    X, y = sample_data
    trainer.train_random_forest(X, y)
    
    # Check if model was trained
    assert 'random_forest' in trainer.models
    
    # Test predictions
    predictions = trainer.models['random_forest'].predict(X)
    assert len(predictions) == len(y)

def test_xgboost_training(trainer, sample_data):
    """Test XGBoost model training."""
    X, y = sample_data
    trainer.train_xgboost(X, y)
    
    # Check if model was trained
    assert 'xgboost' in trainer.models
    
    # Test predictions
    predictions = trainer.models['xgboost'].predict(X)
    assert len(predictions) == len(y)

def test_model_evaluation(trainer, sample_data):
    """Test model evaluation metrics."""
    X, y = sample_data
    trainer.train_random_forest(X, y)
    
    results = trainer.evaluate_models(X, y)
    
    # Check metrics
    assert 'random_forest' in results
    assert 'R2 Score' in results['random_forest']
    assert 'MSE' in results['random_forest']
    assert 'MAE' in results['random_forest']

def test_feature_importance(trainer, sample_data):
    """Test feature importance calculation."""
    X, y = sample_data
    trainer.train_random_forest(X, y)
    
    importance = trainer.calculate_feature_importance(X)
    
    # Check importance calculation
    assert 'random_forest' in importance
    assert len(importance['random_forest']) == len(X.columns)

def test_model_saving_loading(trainer, sample_data, tmp_path):
    """Test model saving and loading."""
    X, y = sample_data
    trainer.train_random_forest(X, y)
    
    # Save models
    save_path = tmp_path / "models"
    trainer.save_models(save_path)
    
    # Check if files were created
    assert os.path.exists(save_path / "random_forest_model.joblib")
    
    # Load models in new trainer
    new_trainer = ModelTrainer()
    new_trainer.load_models(save_path)
    
    # Compare predictions
    original_pred = trainer.predict(X)
    loaded_pred = new_trainer.predict(X)
    np.testing.assert_array_almost_equal(
        original_pred['random_forest'],
        loaded_pred['random_forest']
    )

def test_recommendations(trainer, sample_data):
    """Test recommendation generation."""
    X, y = sample_data
    trainer.train_random_forest(X, y)
    
    recommendations = trainer.get_feature_recommendations(X.iloc[[0]])
    
    # Check recommendation format
    assert len(recommendations) > 0
    assert 'feature' in recommendations[0]
    assert 'impact' in recommendations[0]
    assert 'direction' in recommendations[0]