
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from utils import preprocess_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove classifier
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def extract_feature(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = model(image).squeeze().cpu().numpy()
    return feature / np.linalg.norm(feature)

def extract_all_features(folder):
    features = []
    image_paths = []
    for file in os.listdir(folder):
        if file.endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(folder, file)
            feat = extract_feature(path)
            features.append(feat)
            image_paths.append(path)
    return np.array(features), image_paths
