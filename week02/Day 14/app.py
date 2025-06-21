import streamlit as st
import tensorflow as tf
import numpy as np
from streamlit_drawable_canvas import st_canvas
import cv2
import pandas as pd

# Set page config
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="✍️",
    layout="centered"
)

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('mnist_cnn_model')

def preprocess_image(image):
    """
    Preprocess the drawn image for model prediction
    Args:
        image: Canvas image data
    Returns:
        preprocessed image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    
    # Resize to 28x28
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Invert colors (black background to white)
    inverted = 255 - resized
    
    # Normalize
    normalized = inverted.astype('float32') / 255.0
    
    # Reshape for model input
    return normalized.reshape(1, 28, 28, 1)

def main():
    st.title("Handwritten Digit Recognition")
    st.write("Draw a digit (0-9) in the box below and see the prediction!")
    
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.3)",
        stroke_width=20,
        stroke_color="#FFFFFF",
        background_color="#000000",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    # Load model
    try:
        model = load_model()
    except:
        st.error("Model not found! Please run the training script first.")
        return
    
    # Make prediction when drawing is done
    if canvas_result.image_data is not None:
        # Get the image data
        image = canvas_result.image_data
        
        # Add a predict button
        if st.button("Predict"):
            # Preprocess the image
            processed_image = preprocess_image(image)
            
            # Make prediction
            prediction = model.predict(processed_image, verbose=0)
            predicted_digit = np.argmax(prediction[0])
            confidence = prediction[0][predicted_digit]
            
            # Display results
            st.write("## Prediction Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Predicted Digit", predicted_digit)
            with col2:
                st.metric("Confidence", f"{confidence:.2%}")
            
            # Display probability distribution
            st.write("### Probability Distribution")
            prob_df = pd.DataFrame({
                'Digit': range(10),
                'Probability': prediction[0]
            })
            st.bar_chart(prob_df.set_index('Digit'))

if __name__ == "__main__":
    main() 