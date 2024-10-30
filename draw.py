import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import cv2

# this is the application to prompt the window to draw the item
class DrawScreen:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw a Number")
        
        # Create a canvas widget
        self.canvas = tk.Canvas(root, width=200, height=200, bg='white')
        self.canvas.pack()
        
        # Bind mouse events to canvas
        self.canvas.bind("<B1-Motion>", self.paint)
        
        # Initialize an image to draw on
        self.image = Image.new("RGB", (200, 200), "white")
        self.draw = ImageDraw.Draw(self.image)

        root.bind("d", self.exit)
        

    def paint(self, event):
        # Draw black squares (e.g., 14x14 pixels) on the canvas and image
        x1, y1 = (event.x - 7), (event.y - 7)
        x2, y2 = (event.x + 7), (event.y + 7)
        
        # Draw on the canvas
        self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", outline="black")
        
        # Draw on the image
        self.draw.rectangle([x1, y1, x2, y2], fill="black")

    def get_image(self):
        # Return the drawn image
        return self.image
    
    
    def exit(self, event):
        # Show the scaled size image
        self.root.destroy()
 
