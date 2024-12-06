import pytest
import pandas as pd
import numpy as np
from src.visualizer import Visualizer
import plotly.graph_objects as go

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'Energy': [0.8, 0.6, 0.7],
        'Valence': [0.5, 0.4, 0.6],
        'Danceability': [0.7, 0.5, 0.8],
        'Popularity': [80, 70, 90],
        'Release_Year': [2021, 2022, 2023]
    })

@pytest.fixture
def visualizer():
    """Create Visualizer instance."""
    return Visualizer()

def test_feature_distribution_plots(visualizer, sample_data):
    """Test feature distribution plot creation."""
    features = ['Energy', 'Valence', 'Danceability']
    figs = visualizer.plot_feature_distributions(sample_data, features)
    
    # Check if correct number of plots were created
    assert len(figs) == len(features)
    
    # Check if plots are valid Plotly figures
    for fig in figs:
        assert isinstance(fig, go.Figure)

def test_correlation_matrix(visualizer, sample_data):
    """Test correlation matrix plot creation."""
    fig = visualizer.plot_correlation_matrix(sample_data)
    
    # Check if plot is valid Plotly figure
    assert isinstance(fig, go.Figure)
    
    # Check if heatmap data exists
    assert len(fig.data) > 0
    assert fig.data[0].type == 'heatmap'

def test_feature_importance_plot(visualizer, sample_data):
    """Test feature importance plot creation."""
    importance_df = pd.DataFrame({
        'Feature': ['Energy', 'Valence', 'Danceability'],
        'Importance': [0.5, 0.3, 0.2]
    })
    
    fig = visualizer.plot_feature_importance(importance_df, 'Random Forest')
    
    # Check if plot is valid Plotly figure
    assert isinstance(fig, go.Figure)
    
    # Check if bar plot data exists
    assert len(fig.data) > 0
    assert fig.data[0].type == 'bar'

def test_prediction_scatter(visualizer, sample_data):
    """Test prediction scatter plot creation."""
    y_true = sample_data['Popularity']
    y_pred = y_true + np.random.normal(0, 5, len(y_true))
    
    fig = visualizer.plot_prediction_scatter(y_true, y_pred, 'Random Forest')
    
    # Check if plot is valid Plotly figure
    assert isinstance(fig, go.Figure)
    
    # Check if scatter plot data exists
    assert len(fig.data) >= 2  # Scatter + perfect prediction line
    assert fig.data[0].type == 'scatter'

def test_popularity_trends(visualizer, sample_data):
    """Test popularity trends plot creation."""
    fig = visualizer.plot_popularity_trends(sample_data)
    
    # Check if plot is valid Plotly figure
    assert isinstance(fig, go.Figure)
    
    # Check if line plot data exists
    assert len(fig.data) > 0
    assert fig.data[0].type == 'scatter'

def test_feature_comparison(visualizer, sample_data):
    """Test feature comparison plot creation."""
    fig = visualizer.create_feature_comparison(
        sample_data,
        'Energy',
        'Danceability'
    )
    
    # Check if plot is valid Plotly figure
    assert isinstance(fig, go.Figure)
    
    # Check if scatter plot data exists
    assert len(fig.data) > 0
    assert fig.data[0].type == 'scatter'