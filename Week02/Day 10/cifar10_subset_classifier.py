import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np

def load_and_preprocess_data(selected_classes=[0, 1, 2]):
    """
    Load CIFAR-10 dataset and filter for selected classes.
    Args:
        selected_classes: List of class indices to include (default: [0, 1, 2] for airplane, automobile, bird)
    Returns:
        Preprocessed training and test data with labels
    """
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Filter data for selected classes
    train_mask = np.isin(y_train, selected_classes)
    test_mask = np.isin(y_test, selected_classes)
    
    x_train = x_train[train_mask]
    y_train = y_train[train_mask]
    x_test = x_test[test_mask]
    y_test = y_test[test_mask]
    
    # Normalize pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # One-hot encode labels
    y_train = to_categorical(y_train, num_classes=len(selected_classes))
    y_test = to_categorical(y_test, num_classes=len(selected_classes))
    
    return (x_train, y_train), (x_test, y_test)

def create_model(num_classes):
    """
    Create a simple CNN model.
    Args:
        num_classes: Number of classes to classify
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

def plot_training_history(history):
    """
    Plot training and validation accuracy/loss curves.
    Args:
        history: Training history object from model.fit()
    """
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
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    
    # Define selected classes (airplane, automobile, bird)
    selected_classes = [0, 1, 2]
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data(selected_classes)
    
    # Create data generator for augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    # Create and compile model
    print("Creating model...")
    model = create_model(len(selected_classes))
    model.summary()
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Train model
    print("Training model...")
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=32, subset='training'),
        validation_data=datagen.flow(x_train, y_train, batch_size=32, subset='validation'),
        epochs=50,
        callbacks=[early_stopping]
    )
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Plot training history
    plot_training_history(history)
    
    # Save model
    model.save('cifar10_subset_model.h5')
    print("Model saved as 'cifar10_subset_model.h5'")

if __name__ == "__main__":
    main() 