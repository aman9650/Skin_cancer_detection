#2.15.0
#2.16.1

import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the pre-trained model
# Load the pre-trained model
model = load_model("my_model_skin_cancer - Copy.h5")

# Define class labels
class_labels = [ 'akiec','bcc', 'bkl', 'df','mel', 'nv','vasc']
# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to match the input size of the model
    image = image.resize((224, 224))
    # Convert the image to a numpy array
    image = np.array(image)
    # Normalize the image pixel values
    image = image / 255.0
    # Expand the dimensions to match the input shape of the model
    image = np.expand_dims(image, axis=0)
    return image

# Function to make predictions
def predict(image):
    # Preprocess the image
    processed_image = preprocess_image(image)
    # Make prediction
    predictions = model.predict(processed_image)
    # Get the predicted class label
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_class_index]
    return predicted_class

# Streamlit app
def main():
    st.title("Skin Cancer Classification")
    st.write("Upload an image of a skin lesion for classification")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            # Predict the class label
            prediction = predict(image)
            st.success(f"Predicted class: {prediction}")

if __name__ == "__main__":
    main()
