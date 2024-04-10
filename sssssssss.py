import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image_path, scale_percent=30, blur_kernel=(5, 5)):
    """Resize and blur the image."""
    image = cv2.imread(image_path)
    width = int(image.shape[1] / scale_percent)
    height = int(image.shape[0] / scale_percent)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    blurred_image = cv2.GaussianBlur(resized_image, blur_kernel, 0)
    return blurred_image

def getArea(gray, image, building_contour):
    """Extracts the building area defined by the given contour."""
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [building_contour], -1, (255, 255, 255), -1)
    building = cv2.bitwise_and(image, image, mask=mask)
    canvas = np.ones_like(image) * 255
    canvas[mask == 255] = building[mask == 255]
    return canvas

def extract_largest_contour(image_path, is_preprocess=False, ratio = 0):
    """Extracts the largest contour from the given image."""
    if is_preprocess:
        image = preprocess_image(image_path, scale_percent=ratio, blur_kernel=(7, 7))
    else:
        image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    edges = cv2.Canny(gray, 50, 150)
    dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour, image

def match_and_display_contours(template_path, input_path):
    """Matches contours from template to input image and displays the result."""
    temp = cv2.imread(template_path)
    main = cv2.imread(input_path)

    ratio = temp.shape[0] / main.shape[0]
    template_contour, template_image = extract_largest_contour(template_path, is_preprocess=True, ratio = ratio)
    input_contour, input_image = extract_largest_contour(input_path, 0)

    gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # edges = cv2.Canny(gray, 50, 150)
    dst1 = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, None)

    # dilated_edges_temp = cv2.dilate(dst1, np.ones((3, 3), np.uint8), iterations=1)

    gray_main = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    gray_main = cv2.adaptiveThreshold(gray_main, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # edges_main = cv2.Canny(gray_main, 50, 150)
    dst2 = cv2.morphologyEx(gray_main, cv2.MORPH_OPEN, None)
    # dilated_edges_main = cv2.dilate(dst2, np.ones((3, 3), np.uint8), iterations=1)

    main_cropped = getArea(cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY), input_image, input_contour)
    temp_cropped = getArea(cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY), template_image, template_contour)

    show_with_matplotlib(cv2.cvtColor(dst1, cv2.COLOR_GRAY2BGR), 'ff')
    show_with_matplotlib(cv2.cvtColor(dst2, cv2.COLOR_GRAY2BGR), 'f')
    detector = cv2.ORB_create()
    # keypoints_main, descriptors_main = detector.detectAndCompute(main_cropped, None)
    # keypoints_template, descriptors_template = detector.detectAndCompute(temp_cropped, None)
    # keypoints_main, descriptors_main = detector.detectAndCompute(input_image, None)
    # keypoints_template, descriptors_template = detector.detectAndCompute(template_image, None)
    keypoints_main, descriptors_main = detector.detectAndCompute(dst2, None)
    keypoints_template, descriptors_template = detector.detectAndCompute(gray, None)
    

    # Feature matching
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptors_template, descriptors_main, k=2)

    # Lowe's ratio test
    good_matches = [m for m, n in matches if m.distance < 0.85 * n.distance]

    match_image = cv2.drawMatches(template_image, keypoints_template, input_image, keypoints_main, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    show_with_matplotlib(match_image, 'fffff')

    # Draw matches
    if len(good_matches) > 5:
        src_pts = np.float32([keypoints_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_main[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w = template_image.shape[:2]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        dst = dst.astype(int).reshape(-1, 2)
        input_image = cv2.polylines(input_image, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        x_coords = dst[:, 0]
        y_coords = dst[:, 1]
        top_left = (int(min(x_coords)), int(min(y_coords)))
        bottom_right = (int(max(x_coords)), int(max(y_coords)))

        # Draw a red rectangle
        cv2.rectangle(input_image, top_left, bottom_right, (0, 0, 255), 3)

        show_with_matplotlib(input_image, 'Detected Area with Bounding Box')
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

# Example usage
match_and_display_contours("positive/building1temp.jpg", "building1.jpg")
