# Spotify Song Popularity Predictor 🎵

A Machine Learning web application that predicts and analyzes song popularity on Spotify using audio features.

Try it out: [Live Demo](https://spotifypopularityprediction-s9jzck5sppfkddyg4vkvhh.streamlit.app/)

<div align="center">
  <img src="app/assets/app_preview.png" alt="App Preview" width="800"/>
</div>

## 🎯 Overview

This project uses machine learning to predict a song's potential popularity on Spotify based on its audio characteristics. It combines the power of Random Forest and XGBoost models to provide accurate predictions and actionable insights.

## ✨ Key Features

- **Popularity Prediction**
  - Real-time predictions using multiple ML models
  - Ensemble averaging for improved accuracy
  - Confidence metrics for predictions

- **Feature Analysis**
  - Interactive visualization of audio features
  - Correlation analysis
  - Feature importance insights

- **Optimization Recommendations**
  - Personalized suggestions for improving popularity
  - Impact analysis of feature modifications
  - Easy-to-follow recommendations

## 🚀 Try It Out

Visit the live application: [Spotify Popularity Predictor](https://spotifypopularityprediction-s9jzck5sppfkddyg4vkvhh.streamlit.app/)

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/Cloudyboiii/Spotify_Popularity_Prediction.git
cd Spotify_Popularity_Prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app/app.py
```

## 📊 Features Used

| Feature | Description |
|---------|-------------|
| Energy | Intensity and activity measure |
| Danceability | How suitable for dancing |
| Loudness | Overall loudness in dB |
| Valence | Musical positiveness measure |
| Acousticness | Amount of acoustic sound |
| Instrumentalness | Predicts if a track contains no vocals |
| Tempo | Overall estimated tempo in BPM |
| Speechiness | Presence of spoken words |
| Liveness | Presence of audience in recording |

## 💻 Tech Stack

- **Frontend**: Streamlit
- **Machine Learning**: scikit-learn, XGBoost
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Version Control**: Git

## 🔜 Future Updates

- [ ] Additional ML models integration
- [ ] Batch prediction capability
- [ ] Model retraining functionality
- [ ] Enhanced visualization options
- [ ] API endpoint for predictions

## 👨‍💻 Author

**Badal Gupta**
- GitHub: [Cloudyboiii](https://github.com/Cloudyboiii)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Support

If you found this helpful, please give it a ⭐!
