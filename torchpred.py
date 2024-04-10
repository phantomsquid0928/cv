from torchvision.transforms import functional as F

def predict(image_path, model, device):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = F.to_tensor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(image)

    # Process prediction here
    return prediction

# Load the trained model
model = get_model(num_classes)
model.load_state_dict(torch.load('empire_state_building_detector.pth'))
model.to(device)

# Predict
image_path = 'path_to_test_image.jpg'
prediction = predict(image_path, model, device)
