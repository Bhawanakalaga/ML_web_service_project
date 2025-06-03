from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Attempt to load feature importances file if it exists
feature_importances = None
if os.path.exists('feature_importances.pkl'):
    feature_importances = joblib.load('feature_importances.pkl')
    print("Loaded feature_importances.pkl successfully.")
else:
    print("Warning: feature_importances.pkl not found. Continuing without it.")

# Load model
model = joblib.load('model.pkl')  # Ensure this file is present

@app.route('/')
def home():
    return "ML Web Service is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict([list(data.values())])
    return jsonify({'prediction': prediction.tolist()})

@app.route('/features', methods=['GET'])
def get_feature_importances():
    if feature_importances is None:
        return jsonify({'error': 'Feature importances not available.'}), 404
    return jsonify({'feature_importances': feature_importances.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
