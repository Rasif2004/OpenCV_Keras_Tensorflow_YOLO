# Object Counting using OpenCV
**Author: Muntasir**

## Objective
This exercise demonstrates how to count objects in an image using OpenCV's edge detection and contour finding capabilities. The program processes an input image to identify distinct objects by detecting their edges and contours, then counts and visualizes them.

## Key OpenCV Functions Used

1. `cv2.imread()`: Reads an image from a file
2. `cv2.cvtColor()`: Converts an image from one color space to another (BGR to Grayscale)
3. `cv2.GaussianBlur()`: Applies Gaussian blur to reduce image noise
4. `cv2.Canny()`: Performs edge detection using the Canny algorithm
5. `cv2.findContours()`: Finds contours in a binary image
6. `cv2.drawContours()`: Draws contour outlines on an image
7. `cv2.putText()`: Adds text to an image
8. `cv2.imshow()`: Displays an image in a window
9. `cv2.imwrite()`: Saves an image to a file

## How to Run the Script

1. Ensure you have OpenCV installed:
   ```bash
   pip install opencv-python numpy
   ```

2. Run the script:
   ```bash
   python contour_counter.py
   ```

## Input/Output

- **Input**: `shapes.png` - An image containing distinct shapes/objects
- **Output**: 
  - `result.png` - The processed image with contours drawn and object count displayed
  - Three display windows showing:
    1. Original image
    2. Edge-detected image
    3. Image with contours and object count

## How It Works

1. The image is loaded and converted to grayscale
2. Gaussian blur is applied to reduce noise
3. Canny edge detection identifies object boundaries
4. Contours are found using the edge-detected image
5. Each contour is drawn on a copy of the original image
6. The total number of objects is counted and displayed
7. Results are shown in windows and saved to a file

## Possible Extensions

1. **Filter by Area**: Add minimum/maximum area thresholds to filter out noise or small objects
   ```python
   min_area = 100
   filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
   ```

2. **Shape Classification**: Identify different shapes (circles, squares, triangles) using contour properties
   ```python
   # Example: Detect circles using circularity
   for cnt in contours:
       perimeter = cv2.arcLength(cnt, True)
       area = cv2.contourArea(cnt)
       circularity = 4 * np.pi * area / (perimeter * perimeter)
   ```

3. **Color-based Analysis**: Combine contour detection with color analysis for more accurate object counting

4. **Real-time Processing**: Modify the script to process video input instead of static images

## Notes

- The script assumes objects are distinct and non-overlapping
- Edge detection parameters (Canny thresholds) may need adjustment based on image characteristics
- The current implementation counts all external contours; internal contours (holes) are ignored 