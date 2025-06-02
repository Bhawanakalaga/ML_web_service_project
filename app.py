from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)
model = joblib.load('model.pkl')
feature_importances = joblib.load('feature_importances.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)

    prediction = model.predict(features)[0]
    probs = model.predict_proba(features)[0]
    confidence = max(probs)
    
    return jsonify({
        'prediction': int(prediction),
        'confidence': round(confidence * 100, 2),
        'feature_importances': feature_importances.tolist()
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)




    

