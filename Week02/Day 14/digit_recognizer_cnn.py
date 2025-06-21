import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

def load_and_preprocess_data():
    """
    Load and preprocess the MNIST dataset
    Returns:
        tuple: (x_train, y_train), (x_test, y_test)
    """
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape for CNN input (samples, height, width, channels)
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    # One-hot encode labels
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)

def create_cnn_model():
    """
    Create and return a CNN model for digit recognition
    Returns:
        model: Compiled Keras model
    """
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Dense Layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, x_train, y_train, x_test, y_test):
    """
    Train the CNN model
    Args:
        model: Compiled Keras model
        x_train, y_train: Training data
        x_test, y_test: Test data
    Returns:
        history: Training history
    """
    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=64,
        validation_data=(x_test, y_test),
        verbose=1
    )
    return history

def save_model(model, model_path='mnist_cnn_model'):
    """
    Save the trained model
    Args:
        model: Trained Keras model
        model_path: Path to save the model
    """
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model.save(model_path)
    print(f"Model saved to {model_path}")

def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Create and compile model
    print("Creating CNN model...")
    model = create_cnn_model()
    model.summary()
    
    # Train model
    print("Training model...")
    history = train_model(model, x_train, y_train, x_test, y_test)
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"\nTest accuracy: {test_acc:.4f}")
    
    # Save model
    save_model(model)

if __name__ == "__main__":
    main() 