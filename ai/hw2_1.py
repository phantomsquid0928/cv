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
import time

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
        self.conv3 = nn.Conv2d(32, 48, 3, padding=1)
        self.fc1 = nn.Linear(48 * 12 * 12, 512) #25 25 125 * 125 6 * 6 when 4x4 maxpool
        self.fc2 = nn.Linear(512, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x))) ##
        x = x.view(-1, 48 * 12 * 12) #125 * 125
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((100, 100)), #500 500
    transforms.ToTensor()
])
def evaluate(model, test_loader) :
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == torch.tensor(labels, dtype=torch.long)).sum().item()

    print(f'Accuracy of the model on the test images: {100 * correct / total}%')
def model_maker() :
    start = time.time()
    dataset = BuildingDataSet('data', transform=transform)
    train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True) #4
    test_loader = DataLoader(test_set, batch_size=4, shuffle=False) #4

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = SimpleCNN()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, torch.tensor(labels, dtype=torch.long))
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    end = time.time()
    print(f'learning time : {end - start:.5f} sec')
    evaluate(model, test_loader)
    torch.save(model.state_dict(), 'detector.pth')

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
    model.load_state_dict(torch.load('detector.pth'))

    # image_path = 'positive/building5.jpg'
    model_prediction = predict(image_path, model, transform)

    print(f'Prediction: {"DOG" if model_prediction == 1 else "CAT"}')

path = 'detector.pth'

# svm = cv2.ml.SVM_load('svm_model.xml')
# svm = None
if __name__ == '__main__':
    argv = sys.argv
    if os.path.exists(path):
        if (len(argv) < 2) :
            exit()
        prediction_stage(argv[1])
    else :
        model_maker()
        if (len(argv) < 2) :
            exit()
        prediction_stage(argv[1])