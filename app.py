from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model and feature importances
model = joblib.load('model.pkl')
feature_importances = joblib.load('feature_importances.pkl')

@app.route('/')
def home():
    return render_template('index.html', feature_importances=feature_importances.tolist())

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = round(float(np.max(probabilities)) * 100, 2)
    return jsonify({'prediction': int(prediction), 'confidence': confidence})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)







    

