import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer

def process_input_features(input_data: pd.DataFrame) -> pd.DataFrame:
    """Process input features to match training data format."""
    try:
        processor = DataProcessor()
        
        # Add engineered features
        input_data['Energy_Danceability'] = input_data['Energy'] * input_data['Danceability']
        input_data['Loudness_Energy'] = input_data['Loudness'] * input_data['Energy']
        input_data['Valence_Danceability'] = input_data['Valence'] * input_data['Danceability']
        
        # Add temporal features
        input_data['Release_Year'] = 2024
        input_data['Release_Month'] = 1
        input_data['Days_Since_Release'] = 0
        
        # Ensure required features
        required_features = [
            'Energy', 'Valence', 'Danceability', 'Loudness', 'Acousticness',
            'Tempo', 'Speechiness', 'Liveness', 'Instrumentalness',
            'Release_Year', 'Release_Month', 'Days_Since_Release',
            'Energy_Danceability', 'Loudness_Energy', 'Valence_Danceability'
        ]
        
        for feature in required_features:
            if feature not in input_data.columns:
                input_data[feature] = 0.0
        
        return input_data[required_features]
    except Exception as e:
        st.error(f"Error processing features: {str(e)}")
        return None

def display_analysis(predictions: dict, processed_data: pd.DataFrame, trainer: ModelTrainer, input_data: pd.DataFrame):
    """Display additional analysis section."""
    st.markdown("### Additional Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.checkbox("Show Feature Impact Analysis", key="impact"):
            st.write("#### Feature Impact Analysis")
            baseline_pred = np.mean(list(predictions.values()))
            
            impacts = []
            for feature in input_data.columns:
                test_up = input_data.copy()
                test_up[feature] *= 1.1
                test_up_processed = process_input_features(test_up)
                pred_up = np.mean([
                    model.predict(test_up_processed)[0] 
                    for model in trainer.models.values()
                ])
                impact = abs(pred_up - baseline_pred)
                impacts.append({
                    'Feature': feature,
                    'Current Value': input_data[feature].iloc[0],
                    'Impact Score': impact
                })
            
            impact_df = pd.DataFrame(impacts)
            impact_df = impact_df.sort_values('Impact Score', ascending=False)
            
            st.dataframe(
                impact_df.style.background_gradient(
                    subset=['Impact Score'],
                    cmap='viridis'
                ).format({
                    'Current Value': '{:.3f}',
                    'Impact Score': '{:.3f}'
                })
            )
    
    with col2:
        if st.checkbox("Show Recommendations", key="recs"):
            st.write("#### Recommendations")
            baseline_pred = np.mean(list(predictions.values()))
            test_features = ['Energy', 'Danceability', 'Loudness', 'Valence']
            recommendations = []
            
            for feature in test_features:
                test_up = input_data.copy()
                test_up[feature] *= 1.1
                test_up_processed = process_input_features(test_up)
                pred_up = np.mean([
                    model.predict(test_up_processed)[0] 
                    for model in trainer.models.values()
                ])
                
                test_down = input_data.copy()
                test_down[feature] *= 0.9
                test_down_processed = process_input_features(test_down)
                pred_down = np.mean([
                    model.predict(test_down_processed)[0] 
                    for model in trainer.models.values()
                ])
                
                if pred_up > pred_down and pred_up > baseline_pred:
                    recommendations.append({
                        'feature': feature,
                        'direction': 'increase',
                        'impact': pred_up - baseline_pred
                    })
                elif pred_down > pred_up and pred_down > baseline_pred:
                    recommendations.append({
                        'feature': feature,
                        'direction': 'decrease',
                        'impact': pred_down - baseline_pred
                    })
            
            recommendations.sort(key=lambda x: abs(x['impact']), reverse=True)
            
            if recommendations:
                for i, rec in enumerate(recommendations[:3], 1):
                    st.markdown(
                        f"{i}. {rec['direction'].title()} **{rec['feature']}** "
                        f"(+{rec['impact']:.1f} points)"
                    )
            else:
                st.info("No significant improvements found.")

def render_prediction(trainer: ModelTrainer, settings: Dict[str, Any]) -> None:
    """Render the prediction page."""
    st.header("🎵 Song Popularity Prediction")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            energy = st.slider("Energy", 0.0, 1.0, 0.5)
            danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
            loudness = st.slider("Loudness (dB)", -60.0, 0.0, -10.0)
            valence = st.slider("Valence", 0.0, 1.0, 0.5)
            speechiness = st.slider("Speechiness", 0.0, 1.0, 0.1)
        
        with col2:
            acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
            instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
            tempo = st.slider("Tempo", 0.0, 250.0, 120.0)
            liveness = st.slider("Liveness", 0.0, 1.0, 0.2)
        
        submitted = st.form_submit_button("Predict Popularity")
    
    if submitted:
        input_data = pd.DataFrame({
            'Energy': [energy],
            'Danceability': [danceability],
            'Loudness': [loudness],
            'Valence': [valence],
            'Acousticness': [acousticness],
            'Instrumentalness': [instrumentalness],
            'Tempo': [tempo],
            'Speechiness': [speechiness],
            'Liveness': [liveness]
        })
        
        processed_data = process_input_features(input_data)
        
        if processed_data is not None:
            st.subheader("Predictions")
            cols = st.columns(len(settings['models']))
            predictions = {}
            
            for i, model_name in enumerate(settings['models']):
                model_key = model_name.lower().replace(" ", "_")
                with cols[i]:
                    try:
                        pred = trainer.models[model_key].predict(processed_data)[0]
                        predictions[model_name] = pred
                        st.metric(
                            f"{model_name} Prediction",
                            f"{pred:.1f}",
                            delta=f"{pred - 50:.1f} from average"
                        )
                    except Exception as e:
                        st.error(f"Error with {model_name}: {str(e)}")
            
            if len(predictions) > 1:
                st.metric(
                    "Ensemble Average",
                    f"{np.mean(list(predictions.values())):.1f}",
                    delta=f"{np.mean(list(predictions.values())) - 50:.1f} from average"
                )
            
            display_analysis(predictions, processed_data, trainer, input_data)