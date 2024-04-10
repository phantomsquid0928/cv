import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_with_matplotlib(color_img, title):
    """Helper function to display an image with Matplotlib."""
    img_RGB = color_img[:, :, ::-1]  # Convert BGR to RGB
    plt.figure(figsize=(10, 10))  # Set the figure size to 10x10 inches
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')  # Hide the axis
    plt.show()

# Load image
image = cv2.imread('building1.jpg')  # Replace with your image path

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding
gray_template = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)

# Edge detection
edges = cv2.Canny(gray_template, 50, 150)

# Dilate the edges
dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

# Find contours
contours, _ = cv2.findContours(dilated_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Assuming the largest contour is the building
building_contour = max(contours, key=cv2.contourArea)

# Create an empty mask
mask = np.zeros_like(gray)

# Draw the building contour on the mask
cv2.drawContours(mask, [building_contour], -1, (255, 255, 255), -1)  # -1 to fill the contour

# Extract the building using the mask
building = cv2.bitwise_and(image, image, mask=mask)

# Create a white canvas
canvas = np.ones_like(image) * 255

# Draw the extracted building on the canvas
canvas[mask == 255] = building[mask == 255]

# Display the result
show_with_matplotlib(canvas, 'Building on White Canvas')
