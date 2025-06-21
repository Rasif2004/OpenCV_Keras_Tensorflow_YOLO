# Day 9: MNIST Training with Early Stopping and Metrics Visualization

## Objective
This project demonstrates how to improve neural network training using callbacks and evaluate model performance through metrics visualization. We implement a Multi-Layer Perceptron (MLP) to classify handwritten digits from the MNIST dataset, incorporating early stopping to prevent overfitting and visualizing the training process.

## Implementation Details

### Model Architecture
- Input Layer: Flattened 28x28 MNIST images
- Hidden Layer 1: 128 neurons with ReLU activation
- Hidden Layer 2: 64 neurons with ReLU activation
- Output Layer: 10 neurons with softmax activation (one for each digit)

### Early Stopping Implementation
The `EarlyStopping` callback is implemented with the following parameters:
- `monitor='val_loss'`: Monitors the validation loss
- `patience=3`: Waits for 3 epochs before stopping if no improvement
- `restore_best_weights=True`: Keeps the best model weights
- `verbose=1`: Provides feedback when early stopping is triggered

Benefits of Early Stopping:
- Prevents overfitting by stopping training when validation performance degrades
- Saves computation time by avoiding unnecessary epochs
- Automatically selects the best model weights

### Training Process
1. Data Preprocessing:
   - Normalize pixel values to [0, 1]
   - One-hot encode labels
   - Split training data (80% training, 20% validation)

2. Model Training:
   - Optimizer: Adam
   - Loss Function: Categorical Cross-Entropy
   - Metrics: Accuracy
   - Batch Size: 128
   - Maximum Epochs: 50 (with early stopping)

### Metrics Visualization
The training process is visualized using two plots:
1. Training and Validation Accuracy
2. Training and Validation Loss

These plots help in:
- Identifying overfitting (divergence between training and validation curves)
- Determining optimal training duration
- Understanding model learning dynamics

## Running the Script
1. Ensure required packages are installed:
   ```bash
   pip install tensorflow matplotlib numpy
   ```

2. Run the training script:
   ```bash
   python train_with_earlystop.py
   ```

3. The script will:
   - Train the model with early stopping
   - Display training progress
   - Save training history plots as 'training_history.png'
   - Print final test accuracy and loss

## Results
The model typically achieves:
- Test accuracy: ~97-98%
- Training time: 2-3 minutes (CPU)
- Early stopping usually triggers after 10-15 epochs

## Extensions and Improvements
1. Additional Callbacks:
   - `ModelCheckpoint`: Save best model weights
   - `ReduceLROnPlateau`: Reduce learning rate when plateauing
   - `TensorBoard`: Visualize training in real-time

2. Additional Metrics:
   - Precision and Recall
   - F1-Score
   - Confusion Matrix

3. Model Improvements:
   - Add dropout layers
   - Implement batch normalization
   - Try different architectures (CNN)

## Author
Muntasir

## Next Steps
1. Experiment with different patience values
2. Implement additional callbacks
3. Try different model architectures
4. Add more sophisticated metrics 