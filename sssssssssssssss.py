import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def show_with_matplotlib(color_img, title):
    """Displays an image using Matplotlib."""
    img_RGB = color_img[:, :, ::-1]  # Convert BGR to RGB
    plt.figure(figsize=(10, 10))
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')
    plt.show()

def extract_features(image, feature_extractor):
    """Extracts features from an image using the specified feature extractor."""
    keypoints, descriptors = feature_extractor.detectAndCompute(image, None)
    return descriptors

def sliding_window(image, step_size, window_size):
    """Generates sliding windows over the image."""
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def finder(rf):
    """Uses the trained Random Forest model to detect objects in a new image."""
    image = cv2.imread('negative/notbuilding4.jpg')
    new_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    (winW, winH) = (512, 512)
    stepSize = 32

    feature_extractor = cv2.ORB_create()
    for (x, y, window) in sliding_window(new_image, stepSize, (winW, winH)):
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        window_features = extract_features(window, feature_extractor)
        if window_features is not None:
            window_features = window_features.astype(np.float32)
            _, prediction = rf.predict(window_features)
            if np.round(prediction.mean()) == 1:
                cv2.rectangle(image, (x, y), (x + winW, y + winH), (0, 255, 0), 2)

    show_with_matplotlib(image, 'Detection Result')

def maker():
    """Trains a Random Forest model using the provided dataset."""
    positive_images = ["positive/template.jpg", "positive/template2.jpg", 'positive/template3.png']
    for i in range(1, 10):
        positive_images.append('positive/building' + str(i) + '.jpg')
    negative_images = ['negative/notbuilding' + str(i) + '.jpg' for i in range(1, 8)]

    feature_extractor = cv2.ORB_create()

    train_data = []
    train_labels = []
    for image_path in positive_images + negative_images:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        features = extract_features(img, feature_extractor)
        if features is not None:
            train_data.extend(features)
            label = 1 if image_path in positive_images else 0
            train_labels.extend([label] * len(features))

    train_data = np.array(train_data, dtype=np.float32)
    train_labels = np.array(train_labels)

    rf = cv2.ml.RTrees_create()
    rf.setMaxDepth(10)
    rf.setMinSampleCount(2)
    rf.setRegressionAccuracy(0)
    rf.setUseSurrogates(False)
    rf.setMaxCategories(15)
    rf.setPriors(np.array([1, 1], dtype=np.float32))
    rf.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

    rf.save('rf_model.xml')
    return rf

# Main execution
path = 'rf_model.xml'
if os.path.exists(path):
    rf = cv2.ml.RTrees_load(path)
    finder(rf)
else:
    rf = maker()
    finder(rf)
