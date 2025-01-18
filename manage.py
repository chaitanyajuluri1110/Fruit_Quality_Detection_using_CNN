from flask import Flask, request, jsonify
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load_model('fruit_model_cnn.keras')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['image']
    image = Image.open(file).resize((224, 224))  # Resize to match your model input
    image_array = np.expand_dims(np.array(image) / 255.0, axis=0)
    prediction = model.predict(image_array)
    label = 'Fresh' if prediction[0][0] > 0.5 else 'Rotten'
    return jsonify({'result': label})

if __name__ == '__main__':
    app.run(debug=True)
