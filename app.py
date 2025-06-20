from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from model.predictor import predict_disease
import os
import uuid
import cloudinary.uploader
import cloudinary
import threading
from dotenv import load_dotenv

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load environment variables and configure Cloudinary
load_dotenv()
cloudinary.config(
    cloud_name=os.getenv('CLOUD_NAME'),
    api_key=os.getenv('API_KEY'),
    api_secret=os.getenv('API_SECRET')
)


def upload_to_cloudinary(filepath, result_dict):
    try:
        upload_result = cloudinary.uploader.upload(
            filepath, folder='your_folder_name')
        result_dict['cloudinary_url'] = upload_result.get('secure_url')
    except Exception as e:
        result_dict['cloudinary_error'] = str(e)


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4().hex}_{filename}"
    filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(filepath)

    upload_result = {}
    upload_thread = threading.Thread(
        target=upload_to_cloudinary, args=(filepath, upload_result))
    upload_thread.start()

    try:
        result = predict_disease(filepath)
        print(f"Prediction result:::::: {result}")

        upload_thread.join()

        if 'cloudinary_url' in upload_result:
            result['image_url'] = upload_result['cloudinary_url']
        else:
            result['image_url'] = None
            if 'cloudinary_error' in upload_result:
                result['cloudinary_error'] = upload_result['cloudinary_error']

        return jsonify(result)

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
        print("File deleted")


if __name__ == '__main__':
    app.run(debug=True, port=6000)
