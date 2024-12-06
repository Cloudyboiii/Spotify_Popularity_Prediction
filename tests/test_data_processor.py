import pytest
import pandas as pd
import numpy as np
from src.data_processor import DataProcessor

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'Energy': [0.8, 0.6, 0.7],
        'Valence': [0.5, 0.4, 0.6],
        'Danceability': [0.7, 0.5, 0.8],
        'Loudness': [-5.0, -8.0, -4.0],
        'Acousticness': [0.2, 0.3, 0.1],
        'Tempo': [120, 130, 110],
        'Speechiness': [0.1, 0.2, 0.1],
        'Liveness': [0.2, 0.3, 0.2],
        'Popularity': [80, 70, 90],
        'Release Date': ['2023-01-01', '2023-02-01', '2023-03-01']
    })

@pytest.fixture
def processor():
    """Create DataProcessor instance."""
    return DataProcessor("data/Spotify_data.csv")

def test_data_validation(processor, sample_data):
    """Test data validation functionality."""
    assert processor.validate_audio_features(sample_data)

def test_feature_engineering(processor, sample_data):
    """Test feature engineering process."""
    processed_data = processor.calculate_feature_interactions(sample_data)
    
    # Check if interaction features were created
    assert 'Energy_Danceability' in processed_data.columns
    assert 'Loudness_Energy' in processed_data.columns
    
    # Verify calculations
    np.testing.assert_almost_equal(
        processed_data['Energy_Danceability'],
        sample_data['Energy'] * sample_data['Danceability']
    )

def test_temporal_features(processor, sample_data):
    """Test temporal feature creation."""
    processed_data = processor.calculate_temporal_features(sample_data)
    
    # Check if temporal features were created
    assert 'Release_Year' in processed_data.columns
    assert 'Release_Month' in processed_data.columns
    assert 'Days_Since_Release' in processed_data.columns

def test_preprocessing_pipeline(processor, sample_data):
    """Test entire preprocessing pipeline."""
    X, y = processor.preprocess_data(sample_data)
    
    # Check output shapes
    assert len(X.columns) > len(sample_data.columns)
    assert len(y) == len(sample_data)
    
    # Check if target variable is correct
    np.testing.assert_array_equal(y, sample_data['Popularity'])

def test_train_test_split(processor, sample_data):
    """Test train-test split functionality."""
    X, y = processor.preprocess_data(sample_data)
    X_train, X_test, y_train, y_test = processor.get_train_test_split(X, y, test_size=0.5)
    
    # Check split sizes
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_train) + len(X_test) == len(X)