import tensorflow as tf
from keras.models import load_model

# Load the model once

# Class names — from your model training
class_names = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

def predict_disease(img_path):
    model = load_model('model/tomato_deasese_model.keras')
    # Load and preprocess image
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = img / 255.0

    # Predict
    pred = model.predict(tf.expand_dims(img, axis=0))[0]
    max_index = int(tf.argmax(pred))
    disease = class_names[max_index]
    confidence = float(pred[max_index])
    print(f"Predicted disease: {disease} with confidence: {confidence}")
    return {
        "diseaseName": disease,
        "confidence": confidence
    }
