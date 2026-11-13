"""Model definition, image transforms and loading helpers (shared by train.py and the API)."""
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms

CLASSES = ["NORMAL", "PNEUMONIA"]          # index 0 = normal, 1 = pneumonia
IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]   # ImageNet statistics
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "pneumonia_resnet18.pth"


def build_model(pretrained: bool = True) -> nn.Module:
    """ResNet18 with a new 2-class output layer (transfer learning)."""
    model = None
    if pretrained:
        try:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except Exception as e:  # e.g. no internet to fetch the pretrained weights
            print(f"WARNING: could not load pretrained weights ({e}). Using random weights.")
    if model is None:
        model = models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, len(CLASSES)))
    return model


def get_train_transform():
    """Training images are randomly altered (augmentation) so the model generalises better."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def get_eval_transform():
    """Validation / test / inference: no randomness."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def load_model(path=MODEL_PATH, device="cpu") -> nn.Module:
    checkpoint = torch.load(path, map_location=device)
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval()
