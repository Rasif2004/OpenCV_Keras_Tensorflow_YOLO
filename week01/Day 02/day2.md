# Day 2: Image Watermarking with OpenCV

## Objective
This exercise demonstrates how to add a watermark to an image using OpenCV's Region of Interest (ROI) and bitwise operations. The goal is to place a logo in the bottom-right corner of a main image while preserving the transparency and visibility of both images.

## OpenCV Functions Used

- `cv2.imread()`: Reads an image from a file
- `cv2.cvtColor()`: Converts an image from one color space to another
- `cv2.threshold()`: Applies a fixed-level threshold to each array element
- `cv2.bitwise_and()`: Calculates the per-element bit-wise conjunction of two arrays
- `cv2.bitwise_not()`: Calculates the per-element bit-wise inversion of an array
- `cv2.add()`: Calculates the per-element sum of two arrays
- `cv2.imwrite()`: Saves an image to a specified file
- `cv2.imshow()`: Displays an image in a window

## How to Run the Script

1. Ensure you have OpenCV installed:
   ```bash
   pip install opencv-python numpy
   ```

2. Run the script:
   ```bash
   python watermark.py
   ```

## Input and Output Files

- Input Files:
  - `main.jpg`: The main image to be watermarked
  - `logo.png`: The logo to be used as a watermark

- Output File:
  - `watermarked.jpg`: The final image with the watermark

## Expected Output
The script will:
1. Display the watermarked image in a window
2. Save the watermarked image as `watermarked.jpg`
3. Print a confirmation message when the process is complete

## Extension Ideas

1. **Dynamic Logo Placement**:
   - Modify the script to allow placing the logo in different corners or positions
   - Add command-line arguments to specify the logo position
   - Implement a GUI to interactively select the logo position

2. **Advanced Transparency**:
   - Implement alpha blending for smoother logo integration
   - Add opacity control for the watermark
   - Create a gradient effect for the watermark
