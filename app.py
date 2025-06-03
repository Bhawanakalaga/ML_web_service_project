# app.py

from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

# Try to load feature importances if the file exists
feature_importances = None
if os.path.exists('feature_importances.pkl'):
    feature_importances = joblib.load('feature_importances.pkl')
    print("Feature importances loaded.")
else:
    print("feature_importances.pkl not found. Skipping...")

@app.route('/')
def index():
    return "ML Web Service is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = data['features']  # Expecting: {"features": [[val1, val2, ...]]}
        prediction = model.predict(features)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/feature_importance', methods=['GET'])
def get_feature_importance():
    if feature_importances is not None:
        return jsonify({'feature_importances': feature_importances.tolist()})
    else:
        return jsonify({'error': 'Feature importances not available'}), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use PORT env variable (for Render), else default to 5000
    app.run(host='0.0.0.0', port=port)



    

