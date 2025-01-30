# Rainfall Prediction Model

This project is a simple machine learning model to predict whether it will rain based on weather-related features. The model was trained using a Random Forest Classifier.

## Files

- `data/Rainfall.csv`: The training dataset file.
- `models/rainfall_prediction_model.pkl`: The trained model file.
- `src/Rainfall_Predictor.ipynb`: Script used to train the model.
- `requirements.txt`: Python dependencies.
- `README.md`: Project instructions and usage.

## Installation

### Clone the Repository

1. To get started, clone this repository using:
   ```bash
   git clone https://github.com/1ochaku/Rainfall-Prediction.git
   cd Rainfall-Prediction
   ```

### Requirements

1. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Making Predictions

1. To make predictions using the trained model, use the following Python script:

   ```python
   import pickle
   import pandas as pd

   # Load the trained model and feature names from the pickle file
   with open("models/rainfall_prediction_model.pkl", "rb") as file:
       model_data = pickle.load(file)

   model = model_data['model']
   features = model_data['features']

   # Example input data: Modify with your own values
   input_data = (1014.9, 19.5, 96, 80, 0.0, 40.1, 13.7)  # Replace with your data

   # Make prediction
   input_df = pd.DataFrame([input_data], columns=features)
   prediction = model.predict(input_df)
   print("Prediction result:", "Rainfall" if prediction[0] == 1 else "No Rainfall")
   ```

2. Run the script to get your prediction:
   ```bash
   python predict_script.py
   ```

### Notes:

- Replace the `input_data` tuple with your own weather data for prediction.
- Ensure that the `rainfall_prediction_model.pkl` file is located in the correct path (`models/` folder).
