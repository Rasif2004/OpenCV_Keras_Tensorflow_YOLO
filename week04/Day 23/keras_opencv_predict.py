import cv2
import numpy as np
from tensorflow import keras
import tensorflow as tf

def load_or_create_model():
    """Load or create a simple CNN model for MNIST digit classification."""
    try:
        # Try to load existing model
        model = keras.models.load_model('model.h5')
        print("Loaded existing model from model.h5")
    except:
        # If model doesn't exist, create and train a simple CNN
        print("Creating and training new model...")
        # Load MNIST dataset
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        # Normalize and reshape data
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        
        # Create model
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
        
        # Compile and train
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
        
        # Save the model
        model.save('model.h5')
        print("Model saved as model.h5")
    
    return model

def preprocess_image(frame):
    """Preprocess the captured frame for model input."""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Resize to 28x28 (MNIST input size)
    resized = cv2.resize(gray, (28, 28))
    
    # Normalize pixel values
    normalized = resized.astype('float32') / 255.0
    
    # Reshape for model input (add batch and channel dimensions)
    processed = normalized.reshape(1, 28, 28, 1)
    
    return processed

def main():
    # Load or create the model
    model = load_or_create_model()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Press 's' to capture and predict, 'q' to quit")
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Display the frame
        cv2.imshow('Webcam', frame)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            # Preprocess the frame
            processed_img = preprocess_image(frame)
            
            # Make prediction
            prediction = model.predict(processed_img)
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]
            
            print(f"Predicted digit: {predicted_class} with confidence: {confidence:.2f}")
            
            # Display prediction on frame
            cv2.putText(frame, f"Pred: {predicted_class} ({confidence:.2f})",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Webcam', frame)
            cv2.waitKey(1000)  # Show prediction for 1 second
            
        elif key == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 