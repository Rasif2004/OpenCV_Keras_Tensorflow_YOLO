# Day 8: Building a Multi-Layer Perceptron with Keras

## Objective
This exercise demonstrates the process of building, training, and evaluating a Multi-Layer Perceptron (MLP) using Keras and TensorFlow. We'll work with the MNIST dataset, which is a collection of handwritten digits, to learn the fundamentals of deep learning model development.

## Dataset Overview
The MNIST dataset consists of:
- 60,000 training images
- 10,000 test images
- Each image is 28x28 pixels (784 features when flattened)
- 10 classes (digits 0-9)

## Model Architecture
Our MLP model consists of:
1. Input Layer: 784 neurons (flattened 28x28 images)
2. Hidden Layer 1: 512 neurons with ReLU activation
3. Hidden Layer 2: 256 neurons with ReLU activation
4. Output Layer: 10 neurons with softmax activation (one for each digit)

## Implementation Details

### Data Preprocessing
- Normalize pixel values to range [0, 1]
- Flatten 28x28 images into 1D vectors
- Convert labels to one-hot encoding

### Model Configuration
- Loss Function: Categorical Cross-Entropy
- Optimizer: Adam
- Metrics: Accuracy
- Batch Size: 128
- Epochs: 10
- Validation Split: 20%

## Running the Code
1. Ensure you have the required packages installed:
   ```bash
   pip install tensorflow matplotlib numpy
   ```

2. Run the script:
   ```bash
   python mnist_mlp.py
   ```

The script will:
- Load and preprocess the MNIST dataset
- Create and compile the MLP model
- Train the model for 10 epochs
- Evaluate the model on the test set
- Generate plots of training/validation accuracy and loss

## Expected Output
- Model summary showing layer configurations
- Training progress for each epoch
- Final test accuracy
- Training history plot saved as 'training_history.png'

## Possible Extensions
1. Model Architecture:
   - Add more hidden layers
   - Experiment with different layer sizes
   - Add dropout layers for regularization
   - Try different activation functions

2. Training Process:
   - Implement learning rate scheduling
   - Add early stopping
   - Experiment with different optimizers
   - Try different batch sizes

3. Evaluation:
   - Add confusion matrix visualization
   - Implement k-fold cross-validation
   - Generate sample predictions with visualization

## Author
Muntasir

## Notes
- The model achieves good accuracy on the MNIST dataset, typically around 97-98%
- Training time may vary depending on your hardware
- The generated plots help visualize the model's learning progress 