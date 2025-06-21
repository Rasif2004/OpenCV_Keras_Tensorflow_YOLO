import cv2
import os

def process_image(image_path, output_size=(300, 300)):
    """
    Process a single image: load, resize, convert to grayscale, and save.
    
    Args:
        image_path (str): Path to the input image
        output_size (tuple): Desired output size (width, height)
    
    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return False
            
        # Get the filename without extension
        filename = os.path.splitext(os.path.basename(image_path))[0]
        
        # Resize the image
        resized = cv2.resize(image, output_size)
        
        # Convert to grayscale
        grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Display original image
        cv2.imshow(f"Original - {filename}", resized)
        
        # Display grayscale image
        cv2.imshow(f"Grayscale - {filename}", grayscale)
        
        # Save grayscale image
        output_path = f"{filename}_grayscale.png"
        cv2.imwrite(output_path, grayscale)
        print(f"Saved grayscale image as: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return False

def main():
    # List of images to process
    images = ["example.jpg", "sample.png", "test.bmp"]
    
    # Process each image
    for image in images:
        if not process_image(image):
            print(f"Failed to process {image}")
    
    # Wait for key press and close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 