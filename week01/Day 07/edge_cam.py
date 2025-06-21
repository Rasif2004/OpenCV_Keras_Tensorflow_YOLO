import cv2
import numpy as np
import os
from datetime import datetime

def process_frame(frame):
    """
    Process a single frame to detect edges.
    
    Args:
        frame: Input frame from webcam
        
    Returns:
        edge_frame: Frame with detected edges
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    return edges

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Check if webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Initialize counter for saved images
    save_counter = 1
    
    # Create output directory if it doesn't exist
    output_dir = "edge_frames"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("Press 's' to save frame")
    print("Press 'q' to quit")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Process frame to detect edges
        edge_frame = process_frame(frame)
        
        # Display original and edge-detected frames
        cv2.imshow('Original', frame)
        cv2.imshow('Edge Detection', edge_frame)
        
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        
        # Save frame if 's' is pressed
        if key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{output_dir}/edge_{save_counter}_{timestamp}.png"
            cv2.imwrite(filename, edge_frame)
            print(f"Saved frame as {filename}")
            save_counter += 1
        
        # Break loop if 'q' is pressed
        elif key == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 