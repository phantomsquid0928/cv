import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def show_with_matplotlib(color_img, title):
    """Helper function to display an image with Matplotlib."""
    img_RGB = color_img[:, :, ::-1]  # Convert BGR to RGB
    plt.figure(figsize=(10, 10))  # Set the figure size to 10x10 inches
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')  # Hide the axis
    plt.show()


def extract_features(image, feature_extractor):
    keypoints, descriptors = feature_extractor.detectAndCompute(image, None)
    return descriptors


def sliding_window(image, step_size, window_size):
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def finder() :
    # Load a new image to make predictions
    image = cv2.imread('positive/building1.jpg')
    new_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define window size and step size
    (winW, winH) = (128, 128)  # Window size
    stepSize = 32  # Step size
    # Initialize SIFT or ORB
    feature_extractor = cv2.SIFT_create()  # or cv2.ORB_create()
    # Loop over the sliding window
    for (x, y, window) in sliding_window(new_image, stepSize, (winW, winH)):
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        window_features = extract_features(window, feature_extractor)
        if window_features is not None:
            # window_features = window_features.reshape(1, -1)
            _, prediction = svm.predict(window_features)
            if np.round(prediction.mean()) == 1:
                cv2.rectangle(image, (x, y), (x + winW, y + winH), (0, 255, 0), 2)

    # Display the result
    show_with_matplotlib(image, 'ffff')
def maker():
        
    # Prepare your dataset (paths to your positive and negative images)
    positive_images = ["positive/template.jpg", "positive/template2.jpg", 'positive/template3.png']
    for i in range(1, 10) :
        positive_images.append('positive/building' + str(i) + '.jpg')
    negative_images = []
    for i in range(1, 8) :
        negative_images.append('negative/notbuilding' + str(i) + '.jpg')
    print(negative_images)

    # Initialize SIFT or ORB
    feature_extractor = cv2.SIFT_create()  # or cv2.ORB_create()

    # Initialize training data and labels
    train_data = []
    train_labels = []

    # Process and label images
    for image_path in positive_images + negative_images:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        features = extract_features(img, feature_extractor)

        if features is not None:
            train_data.extend(features)
            label = 1 if image_path in positive_images else 0
            train_labels.extend([label] * len(features))

    # Convert to numpy arrays and train SVM
    train_data = np.float32(train_data)
    train_labels = np.array(train_labels)

    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

    # Save the trained SVM model
    svm.save('svm_model.xml')

# Load the trained SVM model
path = 'svm_model.xml'

# svm = cv2.ml.SVM_load('svm_model.xml')
# svm = None
if os.path.exists(path):
    svm = cv2.ml.SVM_load(path)
    finder()
else :
    maker()
    svm = cv2.ml.SVM_load(path)
    finder()