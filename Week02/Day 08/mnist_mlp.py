import tensorflow as tf
from tensorflow.keras import models, layers, utils, datasets
import matplotlib.pyplot as plt
import numpy as np

def load_and_preprocess_data():
    """
    Load MNIST dataset and preprocess the data
    Returns:
        tuple: (x_train, y_train), (x_test, y_test)
    """
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Flatten the 28x28 images into 1D vectors
    x_train = x_train.reshape((60000, 28 * 28))
    x_test = x_test.reshape((10000, 28 * 28))
    
    # Convert labels to one-hot encoding
    y_train = utils.to_categorical(y_train)
    y_test = utils.to_categorical(y_test)
    
    return (x_train, y_train), (x_test, y_test)

def create_mlp_model():
    """
    Create and compile the MLP model
    Returns:
        model: Compiled Keras model
    """
    # Create Sequential model
    model = models.Sequential([
        # Input layer
        layers.Dense(512, activation='relu', input_shape=(28 * 28,)),
        # Hidden layer
        layers.Dense(256, activation='relu'),
        # Output layer
        layers.Dense(10, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_training_history(history):
    """
    Plot training and validation accuracy/loss
    Args:
        history: Training history object
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
    print("Loading and preprocessing MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Create and compile model
    print("Creating MLP model...")
    model = create_mlp_model()
    
    # Print model summary
    model.summary()
    
    # Train the model
    print("\nTraining the model...")
    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=128,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate the model
    print("\nEvaluating the model...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    
    # Plot training history
    plot_training_history(history)
    print("\nTraining history plot saved as 'training_history.png'")

if __name__ == "__main__":
    main() 