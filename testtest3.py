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

# Load images
main_image = cv2.imread('building2.jpg')  # Replace with your image path
template_image = cv2.imread('positive/template5.jpg')  # Replace with your template image path

temp = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY) 
# Optional: Resize images for resolution normalization
# Change 'new_width' and 'new_height' as needed
# new_height, new_width = temp.shape
# resized_main = cv2.resize(main_image, (new_width, new_height))
# resized_template = cv2.resize(template_image, (new_width, new_height))
# template_image = resized_template

# aspect_ratio = template_image.shape[1] / template_image.shape[0]

# # Define the new width or height
# # For example, let's resize the width to 300 pixels
# new_height = main_image.shape[0]

# # Calculate the new height to maintain the aspect ratio
# new_width = int(new_height * aspect_ratio)

# # Resize the image
# resized_template = cv2.resize(template_image, (new_width, new_height))


# Use resized images for further processing
# gray_main = cv2.cvtColor(resized_main, cv2.COLOR_BGR2GRAY)
# gray_template = cv2.cvtColor(resized_template, cv2.COLOR_BGR2GRAY)
# # # Convert to grayscale
gray_main = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
gray_template = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

# gray_template = cv2.bilateralFilter(gray_template, -1, 20, 30)
gray_main = cv2.equalizeHist(gray_main)
gray_template = cv2.equalizeHist(gray_template)
# # gray_template = cv2.dilate(gray_template, np.ones((5, 5), np.uint8), iterations=1)
# gray_template = cv2.adaptiveThreshold(gray_template, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                cv2.THRESH_BINARY, 21, 16)
# # gray_template = cv2.Canny(gray_template, 50, 150)

# gray_main = cv2.bilateralFilter(gray_main, -1, 20, 30)
# gray_main = cv2.adaptiveThreshold(gray_main, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                cv2.THRESH_BINARY, 11, 15)

# gray_main = cv2.Canny(gray_main, 100, 200)

show_with_matplotlib(cv2.cvtColor(gray_template, cv2.COLOR_GRAY2BGR), 'ffffff')
show_with_matplotlib(cv2.cvtColor(gray_main, cv2.COLOR_GRAY2BGR), 'ffd')
# img_height, img_width = gray_template.shape
# crop_y_start = 0  # Replace with the y-coordinate of the cropping start
# crop_y_end = int(img_height / 1.2)  # Replace with the y-coordinate of the cropping end
# cropped_gray = gray_template[crop_y_start:crop_y_end, :]

# Initialize feature detector
detector = cv2.ORB_create()  # You can also try SURF or ORB

# Find the keypoints and descriptors with SIFT
keypoints_main, descriptors_main = detector.detectAndCompute(gray_main, None)
keypoints_template, descriptors_template = detector.detectAndCompute(gray_template, None)

# Feature matching
matcher = cv2.BFMatcher()
matches = matcher.knnMatch(descriptors_template, descriptors_main, k=2)
# matches = matcher.match(descriptors_main, descriptors_template)
temp_res = cv2.drawKeypoints(gray_template, keypoints_template, None, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
show_with_matplotlib(temp_res, "temres")
# res = cv2.drawMatches(main_image, descriptors_main, template_image, descriptors_template, matcher, template_image)
# show_with_matplotlib(res, "ff")
# Apply Lowe's ratio test to filter good matches
all_matches = [m for mlist in matches for m in mlist]

all_matches = sorted(all_matches, key = lambda x : x.distance)

good_matches = []
# good_matches = all_matches[:50]
for m, n in matches:
    if m.distance < 0.8 * n.distance:
        good_matches.append(m) 

match_image = cv2.drawMatches(gray_template, keypoints_template, main_image, keypoints_main, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
show_with_matplotlib(match_image, 'fffff')

# Find homography and draw matches
if len(good_matches) > 4:
    src_pts = np.float32([keypoints_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_main[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Get the dimensions of the template image
    h, w = template_image.shape[:2]

    corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    projected_corners = cv2.perspectiveTransform(corners, matrix)

    # Draw a red rectangle
    projected_corners = projected_corners.astype(int).reshape(-1, 2)
    detected_img = cv2.polylines(main_image, [projected_corners], True, (0, 0, 255), 3, cv2.LINE_AA)

    x_coords = projected_corners[:, 0]
    y_coords = projected_corners[:, 1]
    top_left = (int(min(x_coords)), int(min(y_coords)))
    bottom_right = (int(max(x_coords)), int(max(y_coords)))

    # Draw a red rectangle
    cv2.rectangle(detected_img, top_left, bottom_right, (255, 0, 0), 3)
    # Project the corners ofnt32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)
else:
    print("Not enough matches found - %d/%d" % (len(good_matches), 4))
    detected_img = main_image


# Display the result
show_with_matplotlib(detected_img, 'fff')
# cv2.imshow('Detected Building', detected_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
