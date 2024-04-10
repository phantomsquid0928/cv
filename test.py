import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_with_matplotlib(color_img, title, pos):
    """Helper function to display multiple images with Matplotlib."""
    img_RGB = color_img[:, :, ::-1]  # BGR to RGB
    ax = plt.subplot(2, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')

def find_intersections(lines):
    intersections = []
    for i, line1 in enumerate(lines):
        for line2 in lines[i+1:]:
            rho1, theta1 = line1[0]
            rho2, theta2 = line2[0]
            # Check if the lines are not too parallel by checking if their angle is not within 10 degrees
            if np.abs(theta1 - theta2) > np.deg2rad(10):
                A = np.array([
                    [np.cos(theta1), np.sin(theta1)],
                    [np.cos(theta2), np.sin(theta2)]
                ])
                b = np.array([[rho1], [rho2]])
                # Find the intersection point
                intersection = np.linalg.solve(A, b)
                if np.all(np.isfinite(intersection)):
                    intersections.append((intersection[0][0], intersection[1][0]))
    return intersections

def filter_extreme_points(points, img_dim):
    # Filter points to keep only those that are within the image boundaries
    return [point for point in points if 0 <= point[0] < img_dim[1] and 0 <= point[1] < img_dim[0]]

# Load the image
img = cv2.imread('checker1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Binarize the grayscale image
_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# Detect edges using Canny
edges = cv2.Canny(binary, 50, 150, apertureSize=3)

# Detect lines using Hough Transform
lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

# Show the intermediate results
plt.figure(figsize=(15, 15))
show_with_matplotlib(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 'Grayscale', 1)
show_with_matplotlib(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), 'Binary', 2)
show_with_matplotlib(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 'Edges', 3)

# If lines are detected, draw them on a copy of the original image for visualization
if lines is not None:
    img_with_lines = img.copy()
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
    show_with_matplotlib(img_with_lines, 'Hough Lines', 4)

# Find intersections between lines
intersections = find_intersections(lines)

# Filter out points that are outside the image
intersections = filter_extreme_points(intersections, img.shape)

# Find the convex hull of the intersections
if intersections:
    hull_points = cv2.convexHull(np.array(intersections, dtype=np.float32))
    hull_points = np.squeeze(hull_points)

    # Draw the convex hull
    img_with_hull = img.copy()
    for point in hull_points:
        cv2.circle(img_with_hull, tuple(int(val) for val in point), 5, (0, 255, 0), -1)

    hull_points = np.append(hull_points, [hull_points[0]], axis=0)  # Close the loop
    for i in range(len(hull_points) - 1):
        cv2.line(img_with_hull, tuple(int(val) for val in hull_points[i]), tuple(int(val) for val in hull_points[i+1]), (255, 0, 0), 2)

    show_with_matplotlib(img_with_hull, 'Convex Hull', 5)

plt.show()
