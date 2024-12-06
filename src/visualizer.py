import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging
from .utils import load_config
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score

class Visualizer:
    """Create visualizations for Spotify song data analysis."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize Visualizer with configuration."""
        self.config = load_config(config_path)
        self.logger = logging.getLogger(__name__)
        self.plot_config = self.config.get('visualization', {})
        
        # Set default colors
        self.colors = {
            'primary': "#1DB954",    # Spotify green
            'secondary': "#191414",  # Spotify black
            'accent': "#1ed760",     # Spotify bright green
            'background': "#FFFFFF", # White
            'text': "#191414"        # Black
        }

    def plot_feature_distributions(
        self,
        data: pd.DataFrame,
        features: List[str]
    ) -> List[go.Figure]:
        """Create distribution plots for selected features."""
        try:
            figs = []
            for feature in features:
                # Calculate statistics
                mean_val = data[feature].mean()
                median_val = data[feature].median()
                std_val = data[feature].std()
                
                # Create histogram with KDE
                fig = go.Figure()
                
                # Add histogram
                fig.add_trace(go.Histogram(
                    x=data[feature],
                    name='Distribution',
                    nbinsx=30,
                    marker_color=self.colors['primary']
                ))
                
                # Add KDE
                kde_x = np.linspace(data[feature].min(), data[feature].max(), 100)
                kde = stats.gaussian_kde(data[feature].dropna())
                kde_y = kde(kde_x) * len(data[feature]) * (data[feature].max() - data[feature].min()) / 30
                
                fig.add_trace(go.Scatter(
                    x=kde_x,
                    y=kde_y,
                    name='Density',
                    line=dict(color=self.colors['accent'])
                ))
                
                # Add mean and median lines
                fig.add_vline(x=mean_val, line_dash="dash", line_color=self.colors['accent'],
                            annotation_text=f"Mean: {mean_val:.2f}")
                fig.add_vline(x=median_val, line_dash="dot", line_color=self.colors['secondary'],
                            annotation_text=f"Median: {median_val:.2f}")
                
                # Update layout
                fig.update_layout(
                    title=f'Distribution of {feature}',
                    template=self.plot_config.get('plot_style', {}).get('template', 'plotly_white'),
                    width=self.plot_config.get('plot_style', {}).get('width', 800),
                    height=self.plot_config.get('plot_style', {}).get('height', 500),
                    xaxis_title=feature,
                    yaxis_title='Count',
                    showlegend=True,
                    annotations=[
                        dict(
                            text=f"Std Dev: {std_val:.2f}",
                            x=0.95,
                            y=0.95,
                            xref="paper",
                            yref="paper",
                            showarrow=False,
                            bgcolor="white",
                            bordercolor=self.colors['primary']
                        )
                    ]
                )
                figs.append(fig)
            return figs
        except Exception as e:
            self.logger.error(f"Error creating distribution plots: {str(e)}")
            raise

    def plot_correlation_matrix(self, data: pd.DataFrame) -> go.Figure:
        """Create correlation matrix heatmap."""
        try:
            numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
            corr_matrix = data[numeric_cols].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmin=-1,
                zmax=1,
                text=np.round(corr_matrix, 2),
                texttemplate='%{text:.2f}',
                textfont={"size": 10}
            ))
            
            fig.update_layout(
                title='Feature Correlation Matrix',
                template=self.plot_config.get('plot_style', {}).get('template', 'plotly_white'),
                width=self.plot_config.get('plot_style', {}).get('width', 800),
                height=self.plot_config.get('plot_style', {}).get('height', 500),
                xaxis_showgrid=False,
                yaxis_showgrid=False,
                yaxis_autorange='reversed'
            )
            
            return fig
        except Exception as e:
            self.logger.error(f"Error creating correlation matrix: {str(e)}")
            raise
            
    def create_feature_comparison(
        self,
        data: pd.DataFrame,
        feature1: str,
        feature2: str
    ) -> go.Figure:
        """Create interactive scatter plot comparing two features."""
        try:
            # Calculate correlation between features
            correlation = data[feature1].corr(data[feature2])
            
            # Create the scatter plot
            fig = px.scatter(
                data,
                x=feature1,
                y=feature2,
                color='Popularity',
                title=f'{feature1} vs {feature2}<br><sup>Correlation: {correlation:.3f}</sup>',
                template=self.plot_config.get('plot_style', {}).get('template', 'plotly_white'),
                color_continuous_scale='Viridis',
                hover_data=['Track Name', 'Artists', 'Popularity']
            )
            
            # Add trend line
            z = np.polyfit(data[feature1], data[feature2], 1)
            p = np.poly1d(z)
            x_range = [data[feature1].min(), data[feature1].max()]
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=p(x_range),
                mode='lines',
                name='Trend',
                line=dict(color=self.colors['accent'], dash='dash')
            ))
            
            # Update layout
            fig.update_layout(
                width=self.plot_config.get('plot_style', {}).get('width', 800),
                height=self.plot_config.get('plot_style', {}).get('height', 500),
                coloraxis_colorbar_title='Popularity',
                showlegend=True,
                hovermode='closest'
            )
            
            return fig
        except Exception as e:
            self.logger.error(f"Error creating feature comparison plot: {str(e)}")
            raise

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        model_name: str
    ) -> go.Figure:
        """Create feature importance plot."""
        try:
            fig = go.Figure()
            
            # Add bar plot
            fig.add_trace(go.Bar(
                y=importance_df['Feature'],
                x=importance_df['Importance'],
                orientation='h',
                marker_color=self.colors['primary']
            ))
            
            # Update layout
            fig.update_layout(
                title=f'Feature Importance ({model_name})',
                template=self.plot_config.get('plot_style', {}).get('template', 'plotly_white'),
                width=self.plot_config.get('plot_style', {}).get('width', 800),
                height=self.plot_config.get('plot_style', {}).get('height', 500),
                xaxis_title='Importance Score',
                yaxis_title='Feature',
                yaxis={'categoryorder':'total ascending'},
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating feature importance plot: {str(e)}")
            raise

    def plot_prediction_scatter(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        model_name: str
    ) -> go.Figure:
        """Create scatter plot of predicted vs actual values."""
        try:
            # Calculate metrics
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            
            # Create figure
            fig = go.Figure()
            
            # Add scatter plot
            fig.add_trace(go.Scatter(
                x=y_true,
                y=y_pred,
                mode='markers',
                name='Predictions',
                marker=dict(
                    color=self.colors['primary'],
                    size=8,
                    opacity=0.6
                ),
                hovertemplate=(
                    "Actual: %{x:.1f}<br>" +
                    "Predicted: %{y:.1f}<br>"
                )
            ))
            
            # Add perfect prediction line
            min_val = min(min(y_true), min(y_pred))
            max_val = max(max(y_true), max(y_pred))
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color=self.colors['accent'], dash='dash')
            ))
            
            fig.update_layout(
                title=f'Predicted vs Actual Popularity ({model_name})<br>'
                      f'<sup>R² = {r2:.3f}, RMSE = {rmse:.3f}</sup>',
                template=self.plot_config.get('plot_style', {}).get('template', 'plotly_white'),
                width=self.plot_config.get('plot_style', {}).get('width', 800),
                height=self.plot_config.get('plot_style', {}).get('height', 500),
                xaxis_title='Actual Popularity',
                yaxis_title='Predicted Popularity',
                showlegend=True,
                hovermode='closest'
            )
            
            return fig
        except Exception as e:
            self.logger.error(f"Error creating prediction scatter plot: {str(e)}")
            raise

    def plot_popularity_trends(self, data: pd.DataFrame) -> go.Figure:
        """Create popularity trends over time plot."""
        try:
            # Calculate yearly statistics
            yearly_stats = data.groupby('Release_Year').agg({
                'Popularity': ['mean', 'std', 'count']
            }).reset_index()
            yearly_stats.columns = ['Release_Year', 'mean_popularity', 'std_popularity', 'count']
            
            # Create figure
            fig = go.Figure()
            
            # Add main trend line
            fig.add_trace(go.Scatter(
                x=yearly_stats['Release_Year'],
                y=yearly_stats['mean_popularity'],
                mode='lines+markers',
                name='Average Popularity',
                line=dict(color=self.colors['primary']),
                error_y=dict(
                    type='data',
                    array=yearly_stats['std_popularity'],
                    visible=True,
                    color=self.colors['secondary']
                ),
                hovertemplate=(
                    "Year: %{x}<br>" +
                    "Average Popularity: %{y:.1f}<br>" +
                    "Standard Deviation: %{error_y.array:.1f}<br>" +
                    "Number of Songs: %{text}<br>"
                ),
                text=yearly_stats['count']
            ))
            
            fig.update_layout(
                title={
                    'text': 'Song Popularity Trends Over Time',
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                template=self.plot_config.get('plot_style', {}).get('template', 'plotly_white'),
                width=self.plot_config.get('plot_style', {}).get('width', 800),
                height=self.plot_config.get('plot_style', {}).get('height', 500),
                xaxis_title='Release Year',
                yaxis_title='Average Popularity',
                showlegend=True,
                hovermode='x unified'
            )
            
            # Add range slider
            fig.update_xaxes(rangeslider_visible=True)
            
            return fig
        except Exception as e:
            self.logger.error(f"Error creating popularity trends plot: {str(e)}")
            raise

    def plot_error_analysis(
        self,
        feature_values: pd.Series,
        errors: pd.Series,
        feature_name: str
    ) -> go.Figure:
        """Create error analysis plot."""
        try:
            fig = go.Figure()
            
            # Add scatter plot
            fig.add_trace(go.Scatter(
                x=feature_values,
                y=errors,
                mode='markers',
                marker=dict(
                    color=self.colors['primary'],
                    opacity=0.6
                ),
                name='Errors',
                hovertemplate=(
                    f"{feature_name}: %{{x:.2f}}<br>" +
                    "Error: %{y:.2f}<br>"
                )
            ))
            
            # Add zero line
            fig.add_hline(
                y=0,
                line_dash="dash",
                line_color=self.colors['secondary'],
                annotation_text="No Error"
            )
            
            # Calculate and add trend line
            z = np.polyfit(feature_values, errors, 1)
            p = np.poly1d(z)
            x_range = [feature_values.min(), feature_values.max()]
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=p(x_range),
                mode='lines',
                name='Trend',
                line=dict(color=self.colors['accent'], dash='dash')
            ))
            
            # Update layout
            fig.update_layout(
                title=f'Error Analysis: {feature_name}',
                template=self.plot_config.get('plot_style', {}).get('template', 'plotly_white'),
                width=self.plot_config.get('plot_style', {}).get('width', 800),
                height=self.plot_config.get('plot_style', {}).get('height', 500),
                xaxis_title=feature_name,
                yaxis_title='Prediction Error',
                showlegend=True,
                hovermode='closest'
            )
            
            return fig
        except Exception as e:
            self.logger.error(f"Error creating error analysis plot: {str(e)}")
            raise