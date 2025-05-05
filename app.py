import numpy as np
import cv2
import tensorflow as tf
import streamlit as st
from tensorflow.keras.preprocessing import image

# Define the preprocess_image function
def preprocess_image(img):
    # Resize image to 224x224
    img = cv2.resize(img, (224, 224))
    
    # Ensure the image has 3 channels (RGB)
    if img.shape[-1] != 3:
        raise ValueError("Image must have 3 channels (RGB).")
    
    # Normalize the image to match ResNet50 input requirements
    img_array = np.array(img) / 255.0
    
    # Add a batch dimension (1, 224, 224, 3)
    img_array = img_array.reshape(1, 224, 224, 3)
    
    return img_array

# Load the model from the specified path
model = tf.keras.models.load_model('./model/resnet50_model.h5')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Streamlit code to upload and process the image
st.title('Pneumonia Classifier')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Patient metadata input fields
age = st.number_input("Age", min_value=0, max_value=120, value=30)  # Example metadata: Age
gender = st.selectbox("Gender", ["Male", "Female"])  # Example metadata: Gender (categorical)

# Convert gender to a numeric value (e.g., one-hot encoding or integer encoding)
gender = 0 if gender == "Male" else 1  # Example: 0 for Male, 1 for Female

if uploaded_file is not None:
    # Read the uploaded image
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Preprocess the image
    try:
        img_array = preprocess_image(img)
    except ValueError as e:
        st.error(f"Error: {e}")
    
    # Preprocess the metadata: Create a feature vector for the metadata
    # Pad the metadata to match the shape (1, 5)
    metadata = np.array([age, gender, 0, 0, 0])  # Add dummy values for the missing features
    
    # Reshape the metadata to match the input shape expected by the model
    metadata = metadata.reshape(1, -1)  # Reshape to (1, 5) if your model expects 5 features
    
    # Make prediction with both inputs: image and metadata
    prediction = model.predict([img_array, metadata])  # Assuming the model expects two inputs

    # Apply a threshold to classify the prediction
    prediction_class = "Pneumonia" if prediction[0][0] > 0.5 else "No Pneumonia"
    
    # Display the result
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write(f"Prediction: {prediction_class} (Probability: {prediction[0][0]:.4f})")
