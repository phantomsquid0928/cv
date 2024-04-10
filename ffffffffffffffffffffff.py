import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_with_matplotlib(color_img, title):
    """ Displays an image with Matplotlib """
    img_RGB = color_img[:, :, ::-1]
    plt.figure(figsize=(10, 10))
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Load images
main_image = cv2.imread('building5.jpg')  # Replace with your path
template_image = cv2.imread('template.jpg')  # Replace with your path

# Convert to grayscale
gray_main = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)

gray_template = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
binary = cv2.adaptiveThreshold(gray_template, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)
gray_template = cv2.Canny(binary, 123, 255)
# Scale down the higher resolution image
# Determine scaling factor
scaling_factor = min(gray_main.shape[0] / gray_template.shape[0], gray_main.shape[1] / gray_template.shape[1])

# Resize if template is larger than main image
if scaling_factor < 1:
    new_size = (int(gray_template.shape[1] * scaling_factor), int(gray_template.shape[0] * scaling_factor))
    gray_template = cv2.resize(gray_template, new_size)

# Initialize SIFT detector
detector = cv2.ORB_create()

# Detect keypoints and descriptors
keypoints_main, descriptors_main = detector.detectAndCompute(gray_main, None)
keypoints_template, descriptors_template = detector.detectAndCompute(gray_template, None)

# Feature matching
matcher = cv2.BFMatcher()
matches = matcher.knnMatch(descriptors_template, descriptors_main, k=2)

# Filter matches using the ratio test
good_matches = [m for m, n in matches if m.distance < 0.86 * n.distance]

# Draw matches
match_image = cv2.drawMatches(template_image, keypoints_template, main_image, keypoints_main, good_matches, None)
show_with_matplotlib(match_image, 'Feature Matches')

# Compute homography and project corners
if len(good_matches) > 4:
    src_pts = np.float32([keypoints_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_main[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Project corners to the main image
    h, w = gray_template.shape
    corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    projected_corners = cv2.perspectiveTransform(corners, matrix)

    # Find extreme points for the rectangle
    x_coords = projected_corners[:, :, 0]
    y_coords = projected_corners[:, :, 1]
    top_left = (int(min(x_coords)[0]), int(min(y_coords)[0]))
    bottom_right = (int(max(x_coords)[0]), int(max(y_coords)[0]))

    # Draw a red rectangle
    detected_img = main_image.copy()
    cv2.rectangle(detected_img, top_left, bottom_right, (0, 0, 255), 3)
else:
    print("Not enough matches found - %d/%d" % (len(good_matches), 4))
    detected_img = main_image.copy()

# Display the result
show_with_matplotlib(detected_img, 'Detected Building')
