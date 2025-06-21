# CIFAR-10 Subset Classification with CNN

## Objective
This project demonstrates how to implement a Convolutional Neural Network (CNN) to classify a subset of the CIFAR-10 dataset. The goal is to learn the process of loading, preprocessing, and training a CNN model on image data using Keras and TensorFlow.

## Dataset Overview
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes. For this project, we focus on three classes:
- Class 0: Airplane
- Class 1: Automobile
- Class 2: Bird

Each class contains 6,000 images, with 5,000 for training and 1,000 for testing.

## Data Preprocessing and Augmentation
The project uses `ImageDataGenerator` for data augmentation, which helps improve model generalization by:
- Random rotation (±20 degrees)
- Random width and height shifts (±20%)
- Random horizontal flips
- Validation split (20% of training data)

The images are normalized to the range [0, 1] by dividing pixel values by 255.

## Model Architecture
The CNN model consists of:
1. Three convolutional blocks, each containing:
   - Conv2D layer with ReLU activation
   - MaxPooling2D layer
2. Flatten layer
3. Dense layer with ReLU activation
4. Dropout layer (0.5)
5. Output layer with softmax activation

## Training Process
The model is trained with:
- Adam optimizer
- Categorical cross-entropy loss
- Early stopping (patience=5)
- Batch size of 32
- Maximum 50 epochs

## How to Run
1. Ensure you have the required packages installed:
   ```bash
   pip install tensorflow matplotlib numpy
   ```

2. Run the script:
   ```bash
   python cifar10_subset_classifier.py
   ```

The script will:
- Load and preprocess the CIFAR-10 dataset
- Train the model with data augmentation
- Save the model as 'cifar10_subset_model.h5'
- Generate training history plots as 'training_history.png'

## Expected Results
The model typically achieves:
- Training accuracy: ~85-90%
- Validation accuracy: ~80-85%
- Test accuracy: ~75-80%

The training curves (saved as 'training_history.png') will show the progression of accuracy and loss during training.

## Possible Extensions
1. **Expand Classes**: Modify the code to use all 10 CIFAR-10 classes
2. **Model Complexity**:
   - Add more convolutional layers
   - Experiment with different filter sizes
   - Try different activation functions
3. **Regularization**:
   - Adjust dropout rate
   - Add L1/L2 regularization
   - Implement batch normalization
4. **Data Augmentation**:
   - Add more augmentation techniques
   - Experiment with different augmentation parameters
5. **Training Process**:
   - Implement learning rate scheduling
   - Try different optimizers
   - Experiment with different batch sizes

## Author
Muntasir

## Notes
- The model uses early stopping to prevent overfitting
- Data augmentation helps improve model generalization
- The training process is monitored using validation metrics
- The model architecture is kept relatively simple for learning purposes 