from roboflow import Roboflow
from ultralytics import YOLO
import supervision as sv
import cv2
import matplotlib.pyplot as plt
import torch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import imghdr
import sys

class BuildingDataSet(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = []
        self.labels = []
        image_exts = ['jpeg', 'jpg', 'bmp', 'png']

        for label in ['positive', 'negative']:
            label_dir = os.path.join(self.directory, label)
            for img_file in os.listdir(label_dir):
                image_path = os.path.join(label_dir, img_file)
                tip = imghdr.what(image_path)
                if tip not in image_exts:
                    os.remove(image_path)
                    continue
                self.images.append(os.path.join(label_dir, img_file))
                self.labels.append(1 if label == 'positive' else 0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 25 * 25, 512) #125 * 125
        self.fc2 = nn.Linear(512, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 25 * 25) #125 * 125
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x



# MODEL EVALUATION CODE, 쓸모 읎슴x
# model.eval()
# correct = 0
# total = 0
# with torch.no_grad():
#     for images, labels in test_loader:
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == torch.tensor(labels, dtype=torch.long)).sum().item()

# print(f'Accuracy of the model on the test images: {100 * correct / total}%')
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((100, 100)), #500 500
    transforms.ToTensor()
])
def model_maker() :
   
    dataset = BuildingDataSet('data', transform=transform)
    train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    # test_l    oader = DataLoader(test_set, batch_size=4, shuffle=False)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = SimpleCNN()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, torch.tensor(labels, dtype=torch.long))
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    torch.save(model.state_dict(), 'building_detector.pth')

def predict(image_path, model, transform):
    model.eval()
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if transform:
        image = transform(image)
    image = image.unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    return predicted.item()

def prediction_stage(image_path) :
    
    # image_path = 'path_to_test_image.jpg'

    model = SimpleCNN()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.load_state_dict(torch.load('building_detector.pth'))

    # image_path = 'positive/building5.jpg'
    model_prediction = predict(image_path, model, transform)

    print(f'Prediction: {"True" if model_prediction == 1 else "False"}')
    return model_prediction

def show_with_matplotlib(color_img, title):
    """Displays an image using Matplotlib."""
    img_RGB = color_img[:, :, ::-1]
    plt.figure(figsize=(10, 10))
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')
    plt.show()


def finder(filename) :
    # rf = Roboflow(api_key="kwjX0ABYSwUbhYirtAyR")
    # project = rf.workspace("test-rawzr").project("empire-state-building-dsgdp")
    # dataset = project.version(1).download("yolov8")
    # model = project.version(1).model
    # torch.save(model,'detector.pt')

    model = torch.load('detector.pt')
    image = cv2.imread(filename)
    result = model.predict(filename, confidence = 30, overlap = 30).json()

    labels = [item["class"] for item in result["predictions"]]

    detections = sv.Detections.from_roboflow(result)

    label_annotator = sv.LabelAnnotator()
    bounding_box_annotator = sv.BoxAnnotator(color = sv.Color(255, 0, 0))



    annotated_image = bounding_box_annotator.annotate(
        scene=image, detections=detections, )
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels)

    show_with_matplotlib(annotated_image, 'ff')
    # sv.plot_image(image=annotated_image, size=(16, 16))

path = 'building_detector.pth'

# svm = cv2.ml.SVM_load('svm_model.xml')
# svm = None
if __name__ == '__main__':
    argv = sys.argv
    if os.path.exists(path):
        if (prediction_stage(argv[1]) == 1) :
            finder(argv[1])
    else :
        model_maker()
        prediction_stage(argv[1])


