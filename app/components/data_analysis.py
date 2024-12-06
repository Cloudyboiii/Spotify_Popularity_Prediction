import streamlit as st
import pandas as pd
import sys
import os

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)


from src.visualizer import Visualizer
from typing import Dict, Any



def render_data_analysis(
    data: pd.DataFrame,
    visualizer: Visualizer,
    settings: Dict[str, Any]
) -> None:
    """Render the data analysis page."""
    
    st.header("📊 Data Analysis")
    
    # Dataset Overview
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Songs", len(data))
    with col2:
        st.metric("Average Popularity", f"{data['Popularity'].mean():.1f}")
    with col3:
        st.metric("Number of Features", len(settings['features']))
    
    # Show raw data sample
    if st.checkbox("Show Raw Data Sample"):
        st.dataframe(data.head())
    
    # Feature Distributions
    st.subheader("Feature Distributions")
    col1, col2 = st.columns(2)
    
    distribution_plots = visualizer.plot_feature_distributions(
        data,
        settings['features']
    )
    
    for i, fig in enumerate(distribution_plots):
        if i % 2 == 0:
            with col1:
                st.plotly_chart(fig, use_container_width=True)
        else:
            with col2:
                st.plotly_chart(fig, use_container_width=True)
    
    # Correlation Analysis
    st.subheader("Feature Correlations")
    correlation_fig = visualizer.plot_correlation_matrix(data)
    st.plotly_chart(correlation_fig, use_container_width=True)
    
    # Show strong correlations
    if settings.get('correlation_threshold'):
        st.subheader("Strong Feature Correlations")
        corr_matrix = data[settings['features']].corr()
        strong_corr = (corr_matrix.abs() > settings['correlation_threshold']).any()
        strong_corr_features = corr_matrix[strong_corr].index.tolist()
        
        if strong_corr_features:
            strong_corr_df = corr_matrix.loc[strong_corr_features, strong_corr_features]
            st.dataframe(strong_corr_df.round(3))
        else:
            st.info("No strong correlations found at the current threshold.")
    
    # Temporal Analysis
    st.subheader("Popularity Trends Over Time")
    trends_fig = visualizer.plot_popularity_trends(data)
    st.plotly_chart(trends_fig, use_container_width=True)
    
    # Feature Comparisons
    st.subheader("Feature Comparisons")
    col1, col2 = st.columns(2)
    with col1:
        feature1 = st.selectbox("Select First Feature", settings['features'])
    with col2:
        feature2 = st.selectbox("Select Second Feature", 
                              [f for f in settings['features'] if f != feature1])
    
    comparison_fig = visualizer.create_feature_comparison(data, feature1, feature2)
    st.plotly_chart(comparison_fig, use_container_width=True)
    
    # Statistical Summary
    if st.checkbox("Show Statistical Summary"):
        st.subheader("Statistical Summary")
        st.dataframe(data[settings['features']].describe().round(3))