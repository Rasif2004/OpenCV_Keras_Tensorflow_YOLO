# OpenCV Drawing Application

## Overview
This project demonstrates how to create a simple interactive drawing application using OpenCV. The application allows users to draw on a blank canvas using mouse interactions, implementing basic drawing tools like freehand drawing and circle drawing.

## Features
- 512×512 pixel white canvas
- Left-click and drag for freehand drawing
- Right-click to draw circles
- Real-time drawing preview
- Save drawing to output.jpg

## OpenCV Functions Used
- `cv2.namedWindow()`: Creates a window to display the canvas
- `cv2.setMouseCallback()`: Registers mouse event handling
- `cv2.imshow()`: Displays the canvas in real-time
- `cv2.waitKey()`: Handles keyboard input
- `cv2.destroyAllWindows()`: Cleans up windows on exit
- `cv2.line()`: Draws lines for freehand drawing
- `cv2.circle()`: Draws circles
- `cv2.imwrite()`: Saves the final drawing

## Mouse Event Handling
The application uses a callback function to handle mouse events:
1. `EVENT_LBUTTONDOWN`: Starts drawing when left mouse button is pressed
2. `EVENT_MOUSEMOVE`: Draws lines while dragging with left button
3. `EVENT_LBUTTONUP`: Stops drawing when left button is released
4. `EVENT_RBUTTONDOWN`: Draws a circle at cursor position

## How to Run
1. Ensure OpenCV is installed: `pip install opencv-python numpy`
2. Run the script: `python draw_app.py`
3. Use the mouse to draw:
   - Left-click and drag for freehand drawing
   - Right-click to draw circles
4. Press 'ESC' to exit and save the drawing

## Input/Output
- Input: None required (starts with blank white canvas)
- Output: Saves drawing as 'output.jpg' in the same directory

## Possible Extensions
1. Add color selection:
   - Implement a color palette
   - Add RGB color picker
2. Add more drawing tools:
   - Rectangle drawing
   - Eraser tool
   - Text tool
3. Add brush customization:
   - Adjustable brush size
   - Different brush styles
4. Add functionality:
   - Undo/Redo
   - Clear canvas
   - Load/Save different drawings
   - Layer support

## Code Structure
The code is organized into two main functions:
1. `draw()`: Handles all mouse events and drawing operations
2. `main()`: Sets up the canvas and runs the main application loop

Global variables are used to maintain the drawing state and canvas between function calls. 