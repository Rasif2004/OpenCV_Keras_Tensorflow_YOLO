# Day 12: Model Serialization and Inference with MNIST CNN

## Goal
This project demonstrates how to save trained deep learning models to disk and reuse them for making predictions. We'll work with a Convolutional Neural Network (CNN) trained on the MNIST dataset, showing both HDF5 and SavedModel serialization formats.

## Model Serialization Formats

### HDF5 Format
- Single file format (`.h5` extension)
- Stores model architecture, weights, and optimizer state
- Compatible with Keras and TensorFlow
- Easy to share and deploy
- Example: `model.save('model.h5')`

### SavedModel Format
- Directory-based format
- TensorFlow's native serialization format
- Includes model architecture, weights, and computation graph
- Better for production deployment
- Example: `model.save('model/')`

## Implementation Details

### Saving Models
```python
# HDF5 format
model.save('mnist_model.h5')

# SavedModel format
model.save('mnist_model/')
```

### Loading Models
```python
from tensorflow.keras.models import load_model

# Load HDF5 model
model = load_model('mnist_model.h5')

# Load SavedModel
model = load_model('mnist_model/')
```

## Running the Script

1. Ensure required packages are installed:
   ```bash
   pip install tensorflow matplotlib numpy
   ```

2. Run the script:
   ```bash
   python save_and_reload_mnist_cnn.py
   ```

The script will:
1. Load and preprocess MNIST data
2. Train a CNN model
3. Save the model to disk
4. Load the saved model
5. Make predictions on test images
6. Display predictions with visualizations

## Example Output
The script will show:
- Training progress
- Model evaluation metrics
- Visualization of test images with predicted and true labels

## Use Cases

1. **Model Deployment**
   - Save trained models for production use
   - Deploy models to different environments

2. **Transfer Learning**
   - Save pre-trained models for fine-tuning
   - Reuse model architectures and weights

3. **Checkpointing**
   - Save model checkpoints during training
   - Resume training from saved states

4. **Model Sharing**
   - Share trained models with others
   - Collaborate on model development

## Author
Muntasir

## Next Steps
- Experiment with different model architectures
- Try different serialization formats
- Implement model versioning
- Add model validation metrics 