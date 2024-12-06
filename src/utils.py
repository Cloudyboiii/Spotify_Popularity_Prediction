import yaml
import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def setup_logging(level: str = "INFO") -> None:
    """Configure logging settings."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def validate_audio_features(data: pd.DataFrame) -> bool:
    """Validate that all required audio features are present and in correct range."""
    required_features = [
        'Energy', 'Valence', 'Danceability', 'Loudness', 
        'Acousticness', 'Tempo', 'Speechiness', 'Liveness'
    ]
    
    # Check if all required features are present
    if not all(feature in data.columns for feature in required_features):
        return False
    
    # Check value ranges
    range_validations = {
        'Energy': (0, 1),
        'Valence': (0, 1),
        'Danceability': (0, 1),
        'Loudness': (-60, 0),
        'Acousticness': (0, 1),
        'Speechiness': (0, 1),
        'Liveness': (0, 1)
    }
    
    for feature, (min_val, max_val) in range_validations.items():
        if not data[feature].between(min_val, max_val).all():
            return False
    
    return True

def calculate_feature_interactions(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate interaction features between audio characteristics."""
    data = data.copy()
    
    # Energy and Danceability interaction
    data['Energy_Danceability'] = data['Energy'] * data['Danceability']
    
    # Loudness and Energy interaction
    data['Loudness_Energy'] = data['Loudness'] * data['Energy']
    
    # Valence and Danceability interaction
    data['Valence_Danceability'] = data['Valence'] * data['Danceability']
    
    return data

def calculate_temporal_features(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate temporal features from release date."""
    data = data.copy()
    
    # Convert release date to datetime if it's not already
    data['Release Date'] = pd.to_datetime(data['Release Date'])
    
    # Extract basic temporal features
    data['Release_Year'] = data['Release Date'].dt.year
    data['Release_Month'] = data['Release Date'].dt.month
    data['Release_Day'] = data['Release Date'].dt.day
    
    # Calculate days since release
    current_date = pd.Timestamp.now()
    data['Days_Since_Release'] = (current_date - data['Release Date']).dt.days
    
    # Create seasonal features
    data['Is_Summer_Release'] = data['Release_Month'].isin([6, 7, 8]).astype(int)
    data['Is_Holiday_Release'] = data['Release_Month'].isin([11, 12]).astype(int)
    
    return data

def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the dataset."""
    data = data.copy()
    
    # Fill numeric columns with median
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
    
    # Fill categorical columns with mode
    categorical_columns = data.select_dtypes(include=['object']).columns
    data[categorical_columns] = data[categorical_columns].fillna(data[categorical_columns].mode().iloc[0])
    
    return data

def remove_outliers(data: pd.DataFrame, columns: list, n_std: float = 3) -> pd.DataFrame:
    """Remove outliers using the z-score method."""
    data = data.copy()
    for column in columns:
        if column in data.columns and data[column].dtype in ['int64', 'float64']:
            z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
            data = data[z_scores < n_std]
    return data

def save_model_metrics(metrics: Dict[str, float], model_name: str, save_path: str) -> None:
    """Save model metrics to a file."""
    metrics_path = Path(save_path) / f"{model_name}_metrics.yaml"
    with open(metrics_path, "w") as f:
        yaml.dump(metrics, f)

def load_model_metrics(model_name: str, metrics_path: str) -> Dict[str, float]:
    """Load model metrics from a file."""
    metrics_file = Path(metrics_path) / f"{model_name}_metrics.yaml"
    with open(metrics_file, "r") as f:
        return yaml.safe_load(f)