"""
Module for detecting objects in a video frame.

This module contains a class for detecting objects in a video frame using a custom CNN model. The class takes a pre
trained model and a image (frame from video stream) and returns if the object is detected in the frame.
"""

import cv2
import numpy as np
import os
from PIL import Image
from threading import Thread
import time
import torch
import torch.nn as nn
from torchvision import transforms
from typing import List

class CustomCNN(nn.Module):
    """
    Class containing the custom CNN model for object detection.

    Args:
        layers (list): A list containing the number of output channels for each layer of the CNN model.
        img_height (int): The height of the input
    
    Attributes:
        layers (nn.ModuleList): A list containing the layers of the CNN model.
        classifier (nn.Sequential): A sequence of layers for the classifier part of the model.
    """
    def __init__(self, layers: List[int], img_height: int = 150):
        super(CustomCNN, self).__init__()
        self.layers = nn.ModuleList()
        input_channels = 3  # Используем 3 канала (RGB)

        for output_channels in layers:
            self.layers.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1))
            self.layers.append(nn.BatchNorm2d(output_channels))  # Добавляем слой BatchNorm
            self.layers.append(nn.ReLU())
            self.layers.append(nn.MaxPool2d(2))
            input_channels = output_channels

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_channels * (img_height // 2**len(layers))**2, 256),  # Первый слой классификатора
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),  # Второй слой классификатора
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)  # Используем 2 выхода для бинарной классификации
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.classifier(x)
        return x


class PhoneDetector:
    """
    Class for detecting objects (right now a phone, hence the name) in a video frame using a custom CNN model.

    Args:
        model_path (str): The path to the pre trained model.
        confidence_threshold (float): The confidence threshold for object detection.
        img_height (int): The height of the input image.
        img_width (int): The width of the input image.
    
    Attributes:
        img_height (int): The height of the input image.
        img_width (int): The width of the input image.
        confidence_threshold (float): The confidence threshold for object detection.
        model (CustomCNN): The custom CNN model for object detection.
        transform (torchvision.transforms.Compose): A series of image transformations to be applied to the input image.
    """
    def __init__(self, model_path: str,confidence_threshold: float = 0.9, img_height: int = 150, img_width: int = 150):
        self.img_height = img_height
        self.img_width = img_width
        self.confidence_threshold = confidence_threshold
        
        # load the model
        self.model = CustomCNN([64, 128, 256], img_height=self.img_height)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        os.makedirs('object', exist_ok=True)

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocesses the input image for object detection.

        Args:
            image (np.ndarray): The input image as a numpy array.

        Returns:
            torch.Tensor: The preprocessed image as a PyTorch tensor.
        """
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = self.transform(image)
        image = image.unsqueeze(0)
        return image

    def detect_phone(self, frame: np.ndarray) -> bool:
        """
        Detects a phone in the input image.

        Args:
            frame (np.ndarray): The input image as a numpy array.

        Returns:
            bool: True if a phone is detected in the image, False otherwise.
        """
        processed_image = self.preprocess_image(frame)
        with torch.no_grad():
            outputs = self.model(processed_image)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        #print(f'Predicted: {predicted.item()}, Confidence: {confidence.item()}')
        return predicted.item() == 1 and confidence.item() >= self.confidence_threshold
