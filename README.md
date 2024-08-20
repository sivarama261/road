# road
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load historical weather data (replace with your data path)
weather_data = pd.read_csv('weather_data.csv')

# Select features for prediction (modify based on your data)
features = ['temperature', 'humidity', 'wind_speed']
X = weather_data[features]
y = weather_data['heat_wave_label']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier (consider other models if needed)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model performance (consider other metrics)
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f'Validation Accuracy: {accuracy}')

# Save the trained model (replace with your desired path)
joblib.dump(model, 'heat_wave_model.pkl')
from flask import Flask, request, jsonify
import joblib

app = Flask(_name_)

# Load trained model
model = joblib.load('heat_wave_model.pkl')

# Define alert threshold (adjust based on model performance)
alert_threshold = 0.5

@app.route('/predict', methods=['POST'])
def predict():
  # Receive JSON data with current weather information (from mobile app)
  req_data = request.get_json()
  temperature = req_data['temperature']
  humidity = req_data['humidity']
  wind_speed = req_data['wind_speed']

  # Predict heat wave probability
  input_data = [[temperature, humidity, wind_speed]]
  probability = model.predict_proba(input_data)[0][1]

  # Determine alert level based on probability threshold
  alert_level = 'High' if probability >= alert_threshold else 'Low'

  # Prepare response for mobile app
  response = {
    'alert_level': alert_level,
    'probability': probability
  }

  return jsonify(response)

if _name_ == '_main_':
  app.run(debug=True)
