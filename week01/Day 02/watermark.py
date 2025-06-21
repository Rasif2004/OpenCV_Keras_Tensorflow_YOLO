import cv2
import numpy as np

def add_watermark(main_image_path, logo_path, output_path):
    """
    Add a watermark to the main image using ROI and bitwise operations.
    
    Args:
        main_image_path (str): Path to the main image
        logo_path (str): Path to the logo image
        output_path (str): Path to save the watermarked image
    """
    # Read the main image and logo
    main_img = cv2.imread(main_image_path)
    logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
    
    if main_img is None or logo is None:
        raise ValueError("Could not load one or both images")
    
    # Get dimensions
    h_main, w_main = main_img.shape[:2]
    h_logo, w_logo = logo.shape[:2]
    
    # Define ROI coordinates (bottom-right corner)
    roi_x = w_main - w_logo - 10  # 10 pixels padding from right
    roi_y = h_main - h_logo - 10  # 10 pixels padding from bottom
    
    # Extract ROI
    roi = main_img[roi_y:roi_y + h_logo, roi_x:roi_x + w_logo]
    
    # Convert logo to grayscale if it's not already
    if len(logo.shape) == 3:
        logo_gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
    else:
        logo_gray = logo
    
    # Create binary mask
    _, mask = cv2.threshold(logo_gray, 240, 255, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)
    
    # Black out the logo area in ROI
    roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    
    # Take only logo region from logo image
    logo_fg = cv2.bitwise_and(logo, logo, mask=mask)
    
    # Combine the background and foreground
    dst = cv2.add(roi_bg, logo_fg)
    
    # Put the combined image back into the main image
    main_img[roi_y:roi_y + h_logo, roi_x:roi_x + w_logo] = dst
    
    # Save the result
    cv2.imwrite(output_path, main_img)
    
    # Display the result
    cv2.imshow('Watermarked Image', main_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    try:
        # File paths
        main_image_path = "main.jpg"
        logo_path = "logo.png"
        output_path = "watermarked.jpg"
        
        # Add watermark
        add_watermark(main_image_path, logo_path, output_path)
        print(f"Watermarked image saved as {output_path}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 