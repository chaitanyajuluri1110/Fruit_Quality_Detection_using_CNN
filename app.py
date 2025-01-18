import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image as PILImage

# Load the trained model
model = load_model('fruit_quality_model.h5')

# Define class labels
class_names = ['freshapples', 'rottenapples', 'freshbanana', 'rottenbanana']

# Function to preprocess and classify the image
def classify_image(image_path, model):
    try:
        # Open the image file
        img = PILImage.open(image_path)

        # Resize the image to the expected size (adjust as necessary)
        img = img.resize((20, 20))  # Change this to the correct size if needed

        # Convert the image to a numpy array and normalize the values to [0, 1]
        img_array = np.array(img) / 255.0

        # Ensure the image has 3 channels (RGB), if it's grayscale, convert it
        if img_array.shape[-1] != 3:
            img_array = np.stack([img_array] * 3, axis=-1)

        # Expand dimensions to match the batch size (1, 20, 20, 3)
        img_array = np.expand_dims(img_array, axis=0)

        # Predict using the model
        predictions = model.predict(img_array)
        
        # Get the predicted class
        predicted_class = class_names[np.argmax(predictions)]
        return predicted_class
    except Exception as e:
        return f"Error in image processing: {e}"

# Streamlit UI
st.title('Fruit Quality Detection')

st.write("Upload an image of a fruit to check its quality (fresh or rotten).")

# File uploader
uploaded_file = st.file_uploader("Choose a fruit image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image.', use_container_width=True)
    
    # Call classify_image to get prediction
    result = classify_image(uploaded_file, model)
    
    st.write(f"Predicted Class: {result}")
