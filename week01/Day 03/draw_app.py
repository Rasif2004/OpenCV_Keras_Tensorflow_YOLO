import cv2
import numpy as np

# Global variables
drawing = False  # True if mouse is pressed
ix, iy = -1, -1  # Starting coordinates
canvas = None    # Canvas to draw on

def draw(event, x, y, flags, param):
    """
    Mouse callback function to handle drawing events
    """
    global ix, iy, drawing, canvas
    
    # Left mouse button down - start drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    
    # Mouse movement - draw line if left button is pressed
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(canvas, (ix, iy), (x, y), (0, 0, 0), 2)
            ix, iy = x, y
    
    # Left mouse button up - stop drawing
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
    
    # Right mouse button - draw circle
    elif event == cv2.EVENT_RBUTTONDOWN:
        cv2.circle(canvas, (x, y), 20, (0, 0, 0), 2)

def main():
    """
    Main function to create and run the drawing application
    """
    global canvas
    
    # Create a white canvas
    canvas = np.ones((512, 512, 3), dtype=np.uint8) * 255
    
    # Create window and set mouse callback
    cv2.namedWindow('Drawing App')
    cv2.setMouseCallback('Drawing App', draw)
    
    print("Drawing App Instructions:")
    print("- Left click and drag to draw")
    print("- Right click to draw a circle")
    print("- Press 'ESC' to exit and save")
    
    while True:
        # Show the canvas
        cv2.imshow('Drawing App', canvas)
        
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        
        # Exit if ESC is pressed
        if key == 27:
            break
    
    # Save the final drawing
    cv2.imwrite('output.jpg', canvas)
    
    # Clean up
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 