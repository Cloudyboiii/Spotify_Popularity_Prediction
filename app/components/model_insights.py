import streamlit as st
import pandas as pd
import numpy as np
from src.model_trainer import ModelTrainer
from src.visualizer import Visualizer
from typing import Dict, Any, List
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def get_model_key(model_name: str) -> str:
    """Convert display model name to internal key."""
    return model_name.lower().replace(" ", "_")

def render_model_insights(
    trainer: ModelTrainer,
    visualizer: Visualizer,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    settings: Dict[str, Any]
) -> None:
    """Render the model insights page."""
    
    st.header("🔍 Model Insights")
    
    # Model Performance Overview
    st.subheader("Model Performance Overview")
    results = trainer.evaluate_models(X_test, y_test)
    
    cols = st.columns(len(settings['models']))
    for i, model_name in enumerate(settings['models']):
        model_key = get_model_key(model_name)
        if model_key in results:
            with cols[i]:
                st.markdown(f"**{model_name} Metrics**")
                metrics = results[model_key]
                st.metric("R² Score", f"{metrics['R2 Score']:.3f}")
                st.metric("RMSE", f"{metrics['RMSE']:.3f}")
                st.metric("MAE", f"{metrics['MAE']:.3f}")
        else:
            with cols[i]:
                st.error(f"No metrics available for {model_name}")
    
    # Detailed Model Analysis
    st.subheader("Detailed Model Analysis")
    
    for model_name in settings['models']:
        model_key = get_model_key(model_name)
        
        if model_key in trainer.models:
            st.markdown(f"### {model_name} Analysis")
            
            # Make predictions
            y_pred = trainer.models[model_key].predict(X_test)
            errors = y_test - y_pred
            
            # Prediction Performance
            st.markdown("#### Prediction Performance")
            fig = visualizer.plot_prediction_scatter(y_test, y_pred, model_name)
            st.plotly_chart(fig, use_container_width=True)
            
            # Error Analysis
            st.markdown("#### Error Analysis")
            error_metrics_cols = st.columns(4)
            
            with error_metrics_cols[0]:
                st.metric("Mean Error", f"{np.mean(errors):.2f}")
            with error_metrics_cols[1]:
                st.metric("Error Std Dev", f"{np.std(errors):.2f}")
            with error_metrics_cols[2]:
                st.metric("Median Error", f"{np.median(errors):.2f}")
            with error_metrics_cols[3]:
                st.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.2f}")
            
            # Error Distribution
            error_df = pd.DataFrame({'Error': errors})
            error_dist_fig = visualizer.plot_feature_distributions(
                error_df,
                features=['Error']
            )[0]
            st.plotly_chart(error_dist_fig, use_container_width=True)
            
            # Feature Importance Analysis
            if st.checkbox(f"Show Feature Importance Analysis for {model_name}", 
                         value=True, key=f"show_importance_{model_key}"):
                st.markdown("#### Feature Importance")
                importance_dict = trainer.calculate_feature_importance(X_test)
                
                if model_key in importance_dict:
                    # Get top N features
                    n_features = settings.get('importance_top_n', 10)
                    importance_df = importance_dict[model_key].head(n_features)
                    
                    # Plot feature importance
                    fig = visualizer.plot_feature_importance(importance_df, model_name)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature importance details
                    if st.checkbox(f"Show detailed feature importance for {model_name}", 
                                 value=False, key=f"feat_imp_{model_key}"):
                        st.dataframe(
                            importance_df.style.format({'Importance': '{:.4f}'})
                            .background_gradient(cmap='Greens', subset=['Importance'])
                        )
                else:
                    st.warning(f"Feature importance not available for {model_name}")
            
            # Prediction Analysis Tools
            if st.checkbox(f"Show prediction explorer for {model_name}", 
                         value=False, key=f"pred_explorer_{model_key}"):
                st.markdown("#### Feature-wise Error Analysis")
                
                # Select features for analysis
                selected_features = st.multiselect(
                    "Select features to analyze",
                    X_test.columns.tolist(),
                    default=X_test.columns.tolist()[:2],
                    key=f"feature_select_{model_key}"
                )
                
                if selected_features:
                    for feature in selected_features:
                        st.markdown(f"**Error Analysis for {feature}**")
                        error_fig = visualizer.plot_error_analysis(
                            X_test[feature],
                            errors,
                            feature
                        )
                        st.plotly_chart(error_fig, use_container_width=True)
                else:
                    st.info("Select features to see their relationship with prediction errors")
        
        else:
            st.error(f"Model {model_name} not found in trained models")
    
    # Model Comparison
    if len(settings['models']) > 1:
        st.subheader("Model Comparison")
        
        comparison_data = []
        for model_name in settings['models']:
            model_key = get_model_key(model_name)
            if model_key in results:
                metrics = results[model_key]
                comparison_data.append({
                    'Model': model_name,
                    'R² Score': metrics['R2 Score'],
                    'RMSE': metrics['RMSE'],
                    'MAE': metrics['MAE']
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            st.markdown("#### Performance Metrics Comparison")
            st.dataframe(
                comparison_df.style.format({
                    'R² Score': '{:.4f}',
                    'RMSE': '{:.4f}',
                    'MAE': '{:.4f}'
                }).background_gradient(cmap='Greens', subset=['R² Score'])
                  .background_gradient(cmap='Reds_r', subset=['RMSE', 'MAE'])
            )
            
            # Download comparison report
            if st.button("Generate Comparison Report"):
                report = generate_comparison_report(
                    comparison_data,
                    results,
                    trainer,
                    X_test,
                    y_test
                )
                st.download_button(
                    "Download Comparison Report",
                    report,
                    file_name="model_comparison_report.txt"
                )

def generate_comparison_report(
    comparison_data: List[Dict],
    results: Dict,
    trainer: ModelTrainer,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> str:
    """Generate a detailed comparison report for all models."""
    report = "Model Comparison Report\n"
    report += "=====================\n\n"
    
    # Performance Metrics
    report += "Performance Metrics:\n"
    report += "-----------------\n"
    for data in comparison_data:
        report += f"\nModel: {data['Model']}\n"
        report += f"R² Score: {data['R² Score']:.4f}\n"
        report += f"RMSE: {data['RMSE']:.4f}\n"
        report += f"MAE: {data['MAE']:.4f}\n"
    
    # Feature Importance Comparison
    report += "\nFeature Importance Comparison:\n"
    report += "---------------------------\n"
    importance_dict = trainer.calculate_feature_importance(X_test)
    
    for model_name in [d['Model'] for d in comparison_data]:
        model_key = get_model_key(model_name)
        if model_key in importance_dict:
            report += f"\n{model_name} Top 10 Features:\n"
            importance_df = importance_dict[model_key].head(10)
            for _, row in importance_df.iterrows():
                report += f"{row['Feature']}: {row['Importance']:.4f}\n"
    
    # Prediction Performance Summary
    report += "\nPrediction Performance Summary:\n"
    report += "----------------------------\n"
    for model_name in [d['Model'] for d in comparison_data]:
        model_key = get_model_key(model_name)
        if model_key in trainer.models:
            y_pred = trainer.models[model_key].predict(X_test)
            errors = y_test - y_pred
            report += f"\n{model_name} Error Statistics:\n"
            report += f"Mean Error: {np.mean(errors):.4f}\n"
            report += f"Error Std Dev: {np.std(errors):.4f}\n"
            report += f"Median Error: {np.median(errors):.4f}\n"
    
    return report