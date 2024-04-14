#2.15.0
#2.16.1
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('final_model .h5')

# Define the class labels
class_labels = ['Actinic keratoses (akiec)', 'Basal cell carcinoma (bcc)', 'Benign keratosis-like lesions (bkl)', 'Dermatofibroma (df)', 'Melanoma (mel)', 'Melanocytic nevi (nv)', 'Vascular lesions (vasc)']

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((65, 65))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to make predictions
def predict(image):
    image = preprocess_image(image)
    predictions = model.predict(image)
    return predictions[0]

# Streamlit app
def main():
    st.title('Skin Cancer Lesion Classification')

    # Upload image
    uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'png'])

    if uploaded_file is not None:
        # Read the uploaded image
        image = Image.open(uploaded_file)

        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Make predictions
        predictions = predict(image)

        # Display the predicted class label
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_labels[predicted_class_index]
        st.write(f'Predicted Class: {predicted_class}')

        # Plot percentage of each class found
        plt.figure(figsize=(10, 8))
        bars = plt.bar(class_labels, predictions * 100, color=plt.cm.viridis(np.linspace(0, 1, len(class_labels))))
        plt.xlabel('Class')
        plt.ylabel('Percentage')
        plt.title('Percentage of Each Class')
        plt.xticks([])
        plt.grid(True)
        legend_labels = [f'{label}: {pred*100:.2f}%' for label, pred in zip(class_labels, predictions)]
        plt.legend(bars, legend_labels, loc='upper right', bbox_to_anchor=(1.3, 1))
        st.pyplot(plt)

# Run the app
if __name__ == '__main__':
    main()
