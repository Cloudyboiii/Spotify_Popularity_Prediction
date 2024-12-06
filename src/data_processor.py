import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
from typing import Tuple, Optional
from pathlib import Path
from datetime import datetime
from .utils import load_config

class DataProcessor:
    """Process and prepare Spotify song data for analysis and modeling."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize DataProcessor with configuration."""
        self.config = load_config(config_path)
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.data = None

    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load and perform initial data processing."""
        try:
            self.data = pd.read_csv(data_path)
            
            # Convert release date to datetime
            self.data['Release Date'] = pd.to_datetime(self.data['Release Date'])
            
            # Create temporal features
            self.data['Release_Year'] = self.data['Release Date'].dt.year
            self.data['Release_Month'] = self.data['Release Date'].dt.month
            self.data['Days_Since_Release'] = (datetime.now() - self.data['Release Date']).dt.days
            
            self.logger.info(f"Successfully loaded and preprocessed data from {data_path}")
            return self.data
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def preprocess_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Perform feature engineering and preprocessing."""
        try:
            if self.data is None:
                raise ValueError("Data not loaded. Call load_data() first.")
            
            # Create interaction features
            self.data['Energy_Danceability'] = self.data['Energy'] * self.data['Danceability']
            self.data['Loudness_Energy'] = self.data['Loudness'] * self.data['Energy']
            self.data['Valence_Danceability'] = self.data['Valence'] * self.data['Danceability']
            
            # Select features
            features = [
                'Energy', 'Valence', 'Danceability', 'Loudness', 'Acousticness',
                'Tempo', 'Speechiness', 'Liveness', 'Instrumentalness',
                'Release_Year', 'Release_Month', 'Days_Since_Release',
                'Energy_Danceability', 'Loudness_Energy', 'Valence_Danceability'
            ]
            
            X = self.data[features]
            y = self.data['Popularity']
            
            # Scale features
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=features,
                index=X.index
            )
            
            return X_scaled, y
            
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {str(e)}")
            raise
    
    def get_train_test_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into training and testing sets."""
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=42
            )
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            self.logger.error(f"Error splitting data: {str(e)}")
            raise

    def transform_new_data(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted scaler."""
        try:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Run preprocess_data first.")
            
            # Create interaction features
            new_data['Energy_Danceability'] = new_data['Energy'] * new_data['Danceability']
            new_data['Loudness_Energy'] = new_data['Loudness'] * new_data['Energy']
            new_data['Valence_Danceability'] = new_data['Valence'] * new_data['Danceability']
            
            # Add temporal features
            current_date = datetime.now()
            new_data['Release_Year'] = current_date.year
            new_data['Release_Month'] = current_date.month
            new_data['Days_Since_Release'] = 0
            
            # Select and order features
            features = [
                'Energy', 'Valence', 'Danceability', 'Loudness', 'Acousticness',
                'Tempo', 'Speechiness', 'Liveness', 'Instrumentalness',
                'Release_Year', 'Release_Month', 'Days_Since_Release',
                'Energy_Danceability', 'Loudness_Energy', 'Valence_Danceability'
            ]
            
            X = new_data[features]
            
            # Scale features
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=features,
                index=X.index
            )
            
            return X_scaled
            
        except Exception as e:
            self.logger.error(f"Error transforming new data: {str(e)}")
            raise