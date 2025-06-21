import cv2
import numpy as np

def nothing(x):
    """Callback function for trackbar"""
    pass

def main():
    # Read the input image in grayscale
    img = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Could not read the image. Make sure 'input.jpg' exists in the current directory.")
        return

    # Create a window
    cv2.namedWindow('Thresholding Demo')

    # Create trackbar for threshold value
    cv2.createTrackbar('Threshold', 'Thresholding Demo', 127, 255, nothing)
    
    # Create trackbar for Gaussian blur
    cv2.createTrackbar('Blur', 'Thresholding Demo', 0, 1, nothing)

    while True:
        # Get current threshold value
        thresh_val = cv2.getTrackbarPos('Threshold', 'Thresholding Demo')
        blur_val = cv2.getTrackbarPos('Blur', 'Thresholding Demo')

        # Apply Gaussian blur if enabled
        if blur_val == 1:
            processed = cv2.GaussianBlur(img, (5, 5), 0)
        else:
            processed = img.copy()

        # Apply thresholding
        _, thresh = cv2.threshold(processed, thresh_val, 255, cv2.THRESH_BINARY)

        # Display original and thresholded images side by side
        combined = np.hstack((img, thresh))
        cv2.imshow('Thresholding Demo', combined)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        
        # Break loop on ESC key
        if key == 27:
            break
        # Save image on 's' key
        elif key == ord('s'):
            cv2.imwrite('thresholded_output.jpg', thresh)
            print("Saved thresholded image as 'thresholded_output.jpg'")

    # Clean up
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 