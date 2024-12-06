import streamlit as st
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import os
from datetime import datetime

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import components
from components.sidebar import create_sidebar
from components.data_analysis import render_data_analysis
from components.model_insights import render_model_insights
from components.prediction import render_prediction

# Import source modules
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
from src.visualizer import Visualizer
from src.utils import load_config, setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Spotify Song Popularity Predictor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def initialize_components(config_path: str = "config.yaml"):
    """Initialize all components with caching."""
    try:
        config = load_config(config_path)
        visualizer = Visualizer(config_path)
        return config, visualizer
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        st.error("Error initializing application components. Please check the configuration.")
        return None, None

@st.cache_data
def load_and_process_data(data_path: str):
    """Load and process data with caching."""
    try:
        processor = DataProcessor()
        processor.load_data(data_path)
        return processor.data, processor
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        st.error(f"Error loading data: {str(e)}")
        return None, None

def main():
    """Main application function."""
    try:
        # Load custom CSS
        css_path = os.path.join(os.path.dirname(__file__), "style", "main.css")
        if os.path.exists(css_path):
            with open(css_path) as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        
        # Initialize components
        config, visualizer = initialize_components()
        if config is None or visualizer is None:
            st.stop()
        
        # Create sidebar and get settings
        page, settings = create_sidebar()
        
        # Title and description
        st.title("🎵 Spotify Song Popularity Predictor")
        st.markdown("""
        This application predicts and analyzes song popularity on Spotify using machine learning models.
        Use the sidebar to navigate between different features and analyses.
        """)
        
        # Check for data and model files
        data_path = "data/Spotify_data.csv"
        if not os.path.exists(data_path):
            st.error(f"Data file not found at {data_path}. Please ensure your data file is in the correct location.")
            st.stop()
        
        # Load and process data
        with st.spinner('Loading and processing data...'):
            data, processor = load_and_process_data(data_path)
            if data is None or processor is None:
                st.stop()
        
        # Initialize trainer and load models
        trainer = ModelTrainer()
        models_path = 'models/saved_models'
        
        if not os.path.exists(models_path):
            st.warning("""
            Models not found. Please run the training script first:
            ```bash
            python train_models.py
            ```
            """)
            st.stop()
        
        try:
            trainer.load_models(models_path)
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            st.stop()
        
        # Render appropriate page
        if page == "Data Analysis":
            render_data_analysis(data, visualizer, settings)
            
        elif page == "Model Insights":
            # Prepare data for model insights
            X, y = processor.preprocess_data()
            X_train, X_test, y_train, y_test = processor.get_train_test_split(X, y)
            render_model_insights(trainer, visualizer, X_test, y_test, settings)
            
        elif page == "Prediction":
            render_prediction(trainer, settings)
        
        # Footer
        st.markdown("---")
        st.markdown(
            f"""
            <div style='text-align: center; color: #666666;'>
                <p>Created with ❤️ using Streamlit and Machine Learning</p>
                <p>Last updated: {datetime.now().strftime('%Y-%m-%d')}</p>
                <p><a href='https://github.com/yourusername/spotify-popularity-predictor' target='_blank'>
                    GitHub Repository</a></p>
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(
            """
            An error occurred while running the application. Please try:
            1. Refreshing the page
            2. Checking if all required files are present
            3. Ensuring models are trained
            """
        )
        if st.checkbox("Show error details"):
            st.exception(e)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.critical(f"Critical application error: {str(e)}")
        st.error("A critical error occurred. Please contact support.")