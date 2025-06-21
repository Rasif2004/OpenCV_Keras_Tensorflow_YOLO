# Interactive Thresholding with OpenCV

## Objective
This exercise demonstrates how to create an interactive thresholding application using OpenCV. The application allows users to dynamically control threshold values using a slider and observe the results in real-time. This is a fundamental concept in image processing that helps in segmenting images based on pixel intensity values.

## OpenCV Functions Used

- `cv2.imread()`: Reads an image from a file
- `cv2.createTrackbar()`: Creates a trackbar (slider) in a window
- `cv2.threshold()`: Applies binary thresholding to an image
- `cv2.GaussianBlur()`: Applies Gaussian blur to reduce noise
- `cv2.imshow()`: Displays an image in a window
- `cv2.waitKey()`: Waits for a key press
- `cv2.destroyAllWindows()`: Closes all windows

## How to Run the Code

1. Make sure you have OpenCV installed:
   ```bash
   pip install opencv-python numpy
   ```

2. Run the script:
   ```bash
   python threshold_slider.py
   ```

## Application Features

- Real-time threshold adjustment using a slider
- Toggle for Gaussian blur application
- Side-by-side display of original and thresholded images
- Save functionality (press 's' to save the thresholded image)
- Exit functionality (press 'ESC' to close the application)

## Understanding the Code

The application works as follows:

1. Loads a grayscale image
2. Creates a window with two trackbars:
   - Threshold slider (0-255)
   - Blur toggle (0-1)
3. Continuously updates the display based on slider positions
4. Applies thresholding in real-time
5. Shows both original and processed images side by side

## Gaussian Blur Effect

The Gaussian blur option helps in:
- Reducing noise in the image
- Creating smoother thresholding results
- Improving edge detection by removing small details

## Possible Extensions

1. Add more thresholding types:
   - Adaptive thresholding
   - Otsu's thresholding
   - Inverted binary thresholding

2. Add more image preprocessing options:
   - Different blur types
   - Contrast adjustment
   - Brightness control

3. Add multiple threshold ranges:
   - Upper and lower threshold values
   - Color-based thresholding

## Author
Muntasir

## Notes
- The application requires a grayscale input image
- The thresholded output is saved as 'thresholded_output.jpg' when pressing 's'
- The application updates in real-time as you move the slider 