import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np

def load_and_preprocess_data():
    """
    Load MNIST dataset and preprocess it by normalizing and one-hot encoding
    """
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalize pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # One-hot encode labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)

def build_model():
    """
    Build a simple MLP model for MNIST classification
    """
    model = Sequential([
        Flatten(input_shape=(28, 28)),  # Flatten 28x28 images
        Dense(128, activation='relu'),  # First hidden layer
        Dense(64, activation='relu'),   # Second hidden layer
        Dense(10, activation='softmax') # Output layer for 10 digits
    ])
    return model

def plot_training_history(history):
    """
    Plot training and validation accuracy/loss
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def main():
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Build and compile model
    model = build_model()
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',    # Monitor validation loss
        patience=3,            # Number of epochs to wait before stopping
        restore_best_weights=True,  # Restore model weights from the epoch with the best value
        verbose=1              # Print message when early stopping is triggered
    )
    
    # Train the model
    history = model.fit(
        x_train, y_train,
        epochs=50,             # Maximum number of epochs
        batch_size=128,
        validation_split=0.2,  # Use 20% of training data for validation
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Plot training history
    plot_training_history(history)

if __name__ == "__main__":
    main() 