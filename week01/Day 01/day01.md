# Image Processing with OpenCV

This Python script demonstrates basic image processing operations using OpenCV. It processes multiple images by resizing them, converting them to grayscale, and saving the results.

## Features

- Loads images in multiple formats (.jpg, .png, .bmp)
- Resizes images to 300×300 pixels
- Converts images to grayscale
- Displays both original and grayscale versions
- Saves grayscale versions with modified filenames

## Requirements

- Python 3.x
- OpenCV (cv2)
- Required Python packages:
  ```
  opencv-python
  ```

## Installation

1. Clone or download this repository
2. Install the required package:
   ```bash
   pip install opencv-python
   ```

## Usage

1. Run the script:
   ```bash
   python image_processor.py
   ```

2. The script will:
   - Display original and grayscale versions of each image
   - Save grayscale versions as:
     - example_grayscale.png
     - sample_grayscale.png
     - test_grayscale.png
   - Wait for a key press before closing all windows

## Code Structure

- `process_image(image_path, output_size=(300, 300))`: Processes a single image
  - Loads the image
  - Resizes to specified dimensions
  - Converts to grayscale
  - Displays both versions
  - Saves grayscale version

- `main()`: Main function that processes all images and manages window display

## Error Handling

The script includes error handling for:
- Failed image loading
- Processing errors
- File I/O operations

## Notes

- Press any key to close all windows after processing
- Grayscale images are saved in PNG format
- Original images are not modified 