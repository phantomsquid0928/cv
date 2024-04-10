import cv2
import numpy as np

def find_building_and_draw_box(target_image_path, reference_image_path):
    # Load images
    img1 = cv2.imread(reference_image_path, 0)  # Reference image (grayscale)
    img2 = cv2.imread(target_image_path, 0)  # Target image (grayscale)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters and matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Filter matches using the Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Homography if enough matches are found
    if len(good_matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find homography matrix
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Get the corners from the reference image
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

        # Project corners into the target image
        dst = cv2.perspectiveTransform(pts, M)

        # Draw bounding box
        x, y, w, h = cv2.boundingRect(dst)
        img2 = cv2.rectangle(cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR), (x, y), (x + w, y + h), (255, 0, 0), 3)

    else:
        print("Not enough matches are found - {}/{}".format(len(good_matches), 10))
        return None

    return img2

# Paths to your images
target_image_path = 'building1.jpg'
reference_image_path = 'template.jpg'

# Find the building and draw the bounding box
result = find_building_and_draw_box(target_image_path, reference_image_path)

if result is not None:
    # Display the image with the bounding box
    cv2.imshow('Found Building', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Building could not be found in the image.")
