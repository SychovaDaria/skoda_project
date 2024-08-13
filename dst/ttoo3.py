import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import time
import os
from raspicam import Raspicam
from threading import Thread

# Define the model
class CustomCNN(nn.Module):
    def __init__(self, layers):
        super(CustomCNN, self).__init__()
        self.layers = nn.ModuleList()
        input_channels = 3  # Using 3 channels (RGB)
        img_height = 150  # Define img_height since it is used in classifier

        for output_channels in layers:
            self.layers.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1))
            self.layers.append(nn.BatchNorm2d(output_channels))  # Add BatchNorm layer
            self.layers.append(nn.ReLU())
            self.layers.append(nn.MaxPool2d(2))
            input_channels = output_channels

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(82944, 512),  # First classifier layer
            #nn.Linear(input_channels * (img_height // 2**len(layers))**2, 512),  # First classifier layer
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),  # Second classifier layer
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)  # Using 2 outputs for binary classification
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.classifier(x)
        return x
    
class PhoneDetector:
    def __init__(self, model_path):
        # Load the model and weights
        self.model = CustomCNN([64, 128])
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()  # Set the model to evaluation mode

        # Parameters
        img_height, img_width = 150, 150

        # Transformations for preprocessing images
        self.transform = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Create directory for saving images if it doesn't exist
        os.makedirs('mobil', exist_ok=True)

    def preprocess_image(self, image):
        """Preprocess the image for the model."""
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = self.transform(image)
        image = image.unsqueeze(0)  # Add batch dimension
        return image

    def detect_phone(self, frame):
        """Detect if there is a phone in the frame."""
        processed_image = self.preprocess_image(frame)
        with torch.no_grad():
            outputs = self.model(processed_image)
            probabilities = torch.softmax(outputs, dim=1)
            print(f"Model outputs: {outputs}")  # Debugging: print raw outputs
            print(f"Probabilities: {probabilities}")  # Debugging: print probabilities
            confidence, predicted = torch.max(probabilities, 1)
            print(f"Predicted: {predicted.item()}, Confidence: {confidence.item()}")  # Debugging: print predicted class and confidence
        return predicted.item() == 1 and confidence.item() >= 0.95

def capture_images(frame):
    """Capture three images when a phone is detected."""
    for i in range(3):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join('mobil', f'phone_detected_{timestamp}.jpg')
        cv2.imwrite(filename, frame)
        print(f'Phone detected! Saved {filename}')
        time.sleep(0.5)  # Wait between captures

