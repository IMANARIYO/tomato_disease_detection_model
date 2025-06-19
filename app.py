from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from model.predictor import predict_disease
import os
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4().hex}_{filename}"
    filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(filepath)

    try:
        result = predict_disease(filepath)
        print(f"Prediction result:::::: {result}")
        return jsonify(result)

    finally:
        # ✅ Always delete the file after prediction (even if there's an error)
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    app.run(debug=True, port=6000)
