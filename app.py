import pandas as pd
import numpy as np
import pickle
import logging
from flask import Flask, render_template, request

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    # Load model and encoder
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    logger.info("Model and label encoder loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or encoder: {e}")
    raise

@app.route('/')
def home():
    logger.info("Home route accessed")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info("Predict route accessed")
        math = float(request.form['math'])
        reading = float(request.form['reading'])
        writing = float(request.form['writing'])
        
        # Create DataFrame with feature names
        features = pd.DataFrame([[math, reading, writing]], 
                               columns=['math score', 'reading score', 'writing score'])
        
        pred = model.predict(features)
        race = le.inverse_transform(pred)[0]
        # Capitalize "group" in the prediction
        race_capitalized = f"Group {race.split(' ')[1]}"
        logger.info(f"Prediction: {race_capitalized}")
        return render_template('index.html', prediction_text=f'Predicted Race/Ethnicity: {race_capitalized}')
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == '__main__':
    app.run(debug=True)