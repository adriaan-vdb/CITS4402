import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import joblib
import cv2
import numpy as np
from skimage.feature import hog
from openpyxl import Workbook

# HOG parameters (use the same ones as in training)
hog_params = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": 'L2-Hys'
}

# Path to model file
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'svm_model.pkl')

# Load the model
model = joblib.load(model_path)

# Function to preprocess the image with aspect ratio preservation
def preprocess_image(image_path, target_size=(128, 64)):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or unable to read")
    
    # Get original dimensions
    h, w = img.shape
    
    # Calculate scale factor to preserve aspect ratio
    scale_factor = max(target_size[0] / h, target_size[1] / w)
    
    # Resize the image
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    img_resized = cv2.resize(img, (new_w, new_h))

    # Center-crop to target size (128, 64)
    start_y = (new_h - target_size[0]) // 2
    start_x = (new_w - target_size[1]) // 2
    img_cropped = img_resized[start_y:start_y + target_size[0], start_x:start_x + target_size[1]]
    
    return img_cropped
    
    # Add padding to maintain the target size (if necessary)
    # top = (target_size[0] - new_h) // 2
    # bottom = target_size[0] - new_h - top
    # left = (target_size[1] - new_w) // 2
    # right = target_size[1] - new_w - left
    
    # Pad the image
    # img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    
    # return img_padded
    # return img_resized

# GUI class
class ImageClassifierGUI:
    def __init__(self, master):
        self.master = master
        master.title("Image Classifier")

        self.label = tk.Label(master, text="Select a folder to begin.")
        self.label.pack()

        self.image_label = tk.Label(master)
        self.image_label.pack()

        self.prediction_label = tk.Label(master, text="", font=("Arial", 16))
        self.prediction_label.pack()

        self.next_button = tk.Button(master, text="Next Image", command=self.next_image, state=tk.DISABLED)
        self.next_button.pack()

        self.load_button = tk.Button(master, text="Load Directory", command=self.load_directory)
        self.load_button.pack()

        self.images = []
        self.index = 0
        self.predictions = []

    def load_directory(self):
        directory = filedialog.askdirectory()
        if not directory:
            return
        self.directory = directory
        self.images = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.images.sort()
        self.index = 0
        self.predictions = []
        self.next_button.config(state=tk.NORMAL)
        self.next_image()

    def next_image(self):
        if self.index >= len(self.images):
            self.save_predictions()
            self.prediction_label.config(text="All images classified. Predictions saved.")
            self.next_button.config(state=tk.DISABLED)
            return

        image_path = os.path.join(self.directory, self.images[self.index])
        
        # Preprocess the image with aspect ratio preservation
        img_padded = preprocess_image(image_path)

        # Extract HOG features
        features = hog(img_padded, **hog_params)

        # Predict
        prediction = model.predict([features])[0]
        label = "Human" if prediction == 1 else "Non-Human"

        # Save prediction
        self.predictions.append((self.images[self.index], int(prediction)))

        # Display image and prediction
        original_image = Image.open(image_path)
        original_image.thumbnail((256, 512))  # Scale down if needed to fit GUI
        tk_img = ImageTk.PhotoImage(original_image)
        self.image_label.config(image=tk_img)
        self.image_label.image = tk_img
        self.prediction_label.config(text=f"Prediction: {label}")

        self.index += 1

    def save_predictions(self):
        wb = Workbook()
        ws = wb.active
        ws.title = "Predictions"
        ws.append(["Filename", "Prediction"])
        for filename, pred in self.predictions:
            ws.append([filename, pred])
        wb.save("predictions.xlsx")


root = tk.Tk()
gui = ImageClassifierGUI(root)
root.mainloop()
