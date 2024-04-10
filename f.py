import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_contours(image_path):
    """Finds contours in the given image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    gray_template = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    edges = cv2.Canny(gray_template, 50, 150)
    dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def get_hu_moments(contour):
    """Calculate Hu Moments for a given contour."""
    moments = cv2.moments(contour)
    huMoments = cv2.HuMoments(moments)
    # Log scale Hu Moments
    for i in range(0,7):
        huMoments[i] = -1 * np.copysign(1.0, huMoments[i]) * np.log10(abs(huMoments[i]))
    return huMoments.flatten()

def compare_hu_moments(huMoments1, huMoments2, threshold=0.1):
    """Compares Hu Moments of two contours."""
    distance = np.linalg.norm(huMoments1 - huMoments2)
    return distance < threshold

def match_partial_shapes(template_path, input_path):
    """Matches partial shapes from the template image in the input image."""
    template_contours = find_contours(template_path)
    input_contours = find_contours(input_path)
    input_image_color = cv2.imread(input_path)

    # Get the Hu Moments for the largest contour in the template
    template_huMoments = get_hu_moments(max(template_contours, key=cv2.contourArea))

    for input_contour in input_contours:
        input_huMoments = get_hu_moments(input_contour)
        if compare_hu_moments(template_huMoments, input_huMoments):
            cv2.drawContours(input_image_color, [input_contour], -1, (0, 255, 0), 3)
            break

    show_with_matplotlib(input_image_color, 'Input Image with Partially Matched Shapes')

def show_with_matplotlib(color_img, title):
    """Displays an image using Matplotlib."""
    img_RGB = color_img[:, :, ::-1]
    plt.figure(figsize=(10, 10))
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Example usage
match_partial_shapes("positive/template6.jpg", "building1.jpg")
