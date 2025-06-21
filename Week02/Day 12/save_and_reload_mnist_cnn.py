import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

def load_and_preprocess_data():
    """Load and preprocess MNIST dataset."""
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape input data to (28, 28, 1)
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    # One-hot encode labels
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    return (x_train, y_train), (x_test, y_test)

def build_cnn_model():
    """Build and compile CNN model."""
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
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

def save_model(model, format='h5'):
    """Save model to disk in specified format."""
    if format == 'h5':
        model.save('mnist_model.h5')
        print("Model saved in HDF5 format as 'mnist_model.h5'")
    else:
        model.save('mnist_model/')
        print("Model saved in SavedModel format in 'mnist_model/' directory")

def load_saved_model(format='h5'):
    """Load model from disk."""
    if format == 'h5':
        return load_model('mnist_model.h5')
    else:
        return load_model('mnist_model/')

def visualize_predictions(model, x_test, y_test, num_samples=5):
    """Visualize model predictions on test images."""
    # Get predictions
    predictions = model.predict(x_test[:num_samples])
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(y_test[:num_samples], axis=1)
    
    # Create figure
    plt.figure(figsize=(12, 4))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        plt.title(f'Pred: {predicted_labels[i]}\nTrue: {true_labels[i]}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # Load and preprocess data
    print("Loading and preprocessing MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Build and train model
    print("\nBuilding and training CNN model...")
    model = build_cnn_model()
    history = model.fit(x_train, y_train, 
                       epochs=5, 
                       batch_size=64, 
                       validation_split=0.2)
    
    # Save model in HDF5 format
    print("\nSaving model...")
    save_model(model, format='h5')
    
    # Load model and verify
    print("\nLoading saved model...")
    loaded_model = load_saved_model(format='h5')
    
    # Evaluate loaded model
    print("\nEvaluating loaded model...")
    test_loss, test_acc = loaded_model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Visualize predictions
    print("\nVisualizing predictions...")
    visualize_predictions(loaded_model, x_test, y_test)

if __name__ == "__main__":
    main() 