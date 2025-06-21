import cv2
import numpy as np

def main():
    # Read the input image
    image = cv2.imread('shapes.png')
    if image is None:
        print("Error: Could not read the image.")
        return

    # Create a copy of the original image for drawing contours
    original = image.copy()

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    # Parameters: (image, kernel size, sigma X)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    # Parameters: (image, threshold1, threshold2)
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edge-detected image
    # Parameters: (image, contour retrieval mode, contour approximation method)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    # Parameters: (image, contours, contour index, color, thickness)
    cv2.drawContours(original, contours, -1, (0, 255, 0), 2)

    # Count the number of objects (contours)
    object_count = len(contours)

    # Add text showing the object count
    # Parameters: (image, text, position, font, scale, color, thickness)
    cv2.putText(original, f'Objects: {object_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the images
    cv2.imshow('Original Image', image)
    cv2.imshow('Edge Detection', edges)
    cv2.imshow('Contours', original)

    # Save the result
    cv2.imwrite('result.png', original)

    # Wait for a key press and close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 