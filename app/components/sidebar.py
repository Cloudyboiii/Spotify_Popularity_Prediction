import streamlit as st
from typing import Tuple, Dict, Any

def create_sidebar() -> Tuple[str, Dict[str, Any]]:
    """Create and manage the application sidebar."""
    
    st.sidebar.header("Navigation")
    
    # Page selection
    page = st.sidebar.radio(
        "Select Page",
        ["Data Analysis", "Model Insights", "Prediction"]
    )
    
    # Feature selection for analysis
    st.sidebar.header("Feature Selection")
    features = st.sidebar.multiselect(
        "Select Features for Analysis",
        ["Energy", "Valence", "Danceability", "Loudness", "Acousticness",
         "Tempo", "Speechiness", "Liveness", "Instrumentalness"],
        default=["Energy", "Danceability", "Valence"]
    )
    
    # Model selection
    model_selection = st.sidebar.multiselect(
        "Select Models",
        ["Random Forest", "XGBoost"],
        default=["Random Forest", "XGBoost"]
    )
    
    # Additional settings based on page
    settings = {}
    settings['features'] = features
    settings['models'] = model_selection
    
    if page == "Data Analysis":
        settings['show_outliers'] = st.sidebar.checkbox("Show Outliers", value=True)
        settings['correlation_threshold'] = st.sidebar.slider(
            "Correlation Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5
        )
        
    elif page == "Model Insights":
        settings['importance_top_n'] = st.sidebar.slider(
            "Top N Important Features",
            min_value=5,
            max_value=20,
            value=10
        )
        settings['show_feature_interactions'] = st.sidebar.checkbox(
            "Show Feature Interactions",
            value=False
        )
        
    elif page == "Prediction":
        settings['show_confidence'] = st.sidebar.checkbox(
            "Show Prediction Confidence",
            value=True
        )
        settings['show_recommendations'] = st.sidebar.checkbox(
            "Show Recommendations",
            value=True
        )
    
    return page, settings