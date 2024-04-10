import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_largest_contour(gray_image):
    """Extract the largest contour from a grayscale image."""
    edged = cv2.Canny(gray_image, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea)

def extract_features(image, contour):
    """Extracts ORB features from the area defined by the given contour."""
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [contour], -1, 255, -1)
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(masked_image, None)
    return keypoints, descriptors

def knn_match_and_draw_box(kp1, kp2, des1, des2, img1, img2):
    """Use KNN matching instead of BF matching and draw bounding box."""
    # Convert descriptors to float32 for FLANN
    des1 = np.float32(des1)
    des2 = np.float32(des2)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.9 * n.distance:
            good_matches.append(m)

    if len(good_matches) > 5:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        h, w = img1.shape[:2]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        img2 = cv2.polylines(img2, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)
        show_with_matplotlib(img2, 'Detected Area with Bounding Box')
    else:
        print("Not enough matches are found - {}/{}".format(len(good_matches), 5))

def show_with_matplotlib(color_img, title):
    """Displays an image using Matplotlib."""
    img_RGB = color_img[:, :, ::-1]
    plt.figure(figsize=(10, 10))
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')
    plt.show()

def find_and_draw_contours(template_path, input_path):
    """Find and draw contours based on the template."""
    template_image = cv2.imread(template_path)
    input_image = cv2.imread(input_path)

    template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
    input_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    template_contour = extract_largest_contour(template_gray)
    input_contour = extract_largest_contour(input_gray)

    kp1, des1 = extract_features(template_image, template_contour)
    kp2, des2 = extract_features(input_image, input_contour)

    knn_match_and_draw_box(kp1, kp2, des1, des2, template_image, input_image)

# Example usage
find_and_draw_contours("positive/building1temp.jpg", "building1.jpg")
