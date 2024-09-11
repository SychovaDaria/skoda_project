import cv2
from enum import Enum
import numpy as np
import os
import time
from typing import List
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

DEFAULT_TRIGGER_DELAY = 0
DEFAULT_NUM_OF_PICTURES = 2
DEFAULT_TIMES_BETWEEN_PICTURES = 1.0
DEFAULT_TIME_TO_RESET = 10.0

class TriggerModes(Enum):
    ALWAYS = 0
    RISING_EDGE = 1
    FALLING_EDGE = 2

class Trigger:
    def __init__(self, camera, folder_name: str, trigger_delay: float = DEFAULT_TRIGGER_DELAY,
                 num_of_pictures: int = 2, time_to_reset: float = 10.0,
                 times_between_pictures: float | List[float] = 1.0,
                 trigger_mode: TriggerModes = TriggerModes.ALWAYS) -> None:
        self.camera = camera
        self.trigger_delay = trigger_delay
        self.time_to_reset = time_to_reset  # 10 seconds of waiting
        self.num_of_pictures = num_of_pictures

        if isinstance(times_between_pictures, (float, int)):
            self.times_between_pictures = [times_between_pictures] * (num_of_pictures - 1)
        else:
            self.times_between_pictures = times_between_pictures

        self.folder_name = folder_name
        self.trigger_mode = trigger_mode
        self.last_trigger_signal = False
        self.capturing = False
        self.delay_number = 0
        self.next_check_time = time.time()  # Timestamp for when to start checking again
        self.first_trigger_time = None  # Initialize first_trigger_time as None
        self.last_trigger_time = None  # Initialize last_trigger_time as None
        self.check_attributes()

    def check_attributes(self) -> None:
        if not (isinstance(self.folder_name, str) and os.path.isdir(self.folder_name)):
            raise ValueError("The folder_name attribute must be a string representing a valid directory path.")
        if not isinstance(self.trigger_mode, TriggerModes):
            raise ValueError("The trigger_mode attribute must be an instance of the TriggerModes enum.")

    def process_trigger_signal(self, trigger_signal: bool, confidence: float, img: np.ndarray) -> None:
        # Check if enough time has passed to resume detection
        if time.time() < self.next_check_time:
            return  # Do nothing, wait for the next check time

        # If an object is detected with sufficient confidence and capturing is not in progress
        if trigger_signal and confidence >= 0.53 and not self.capturing:
            print("Object detected with sufficient confidence. Starting capture.")
            self.capturing = True  # Start capturing photos
            self.first_trigger_time = time.time()  # Set the timestamp for this capture session
            self.last_trigger_time = self.first_trigger_time  # Initialize last_trigger_time
            self.delay_number = 0
            self.capture_imgs(img)

        # If capturing is ongoing, continue capturing images
        if self.capturing:
            if self.delay_number < self.num_of_pictures:
                wait_time = self.times_between_pictures[self.delay_number - 1] if self.delay_number > 0 else self.trigger_delay
                if self.wait_for_delay(self.last_trigger_time, wait_time):
                    self.last_trigger_time = time.time()  # Update last_trigger_time after each capture
                    self.capture_imgs(img)
            else:
                # After taking two photos, stop capturing and set the next check time (10 seconds later)
                self.capturing = False
                self.next_check_time = time.time() + self.time_to_reset  # 10 seconds of pause
                print(f"Pausing for {self.time_to_reset} seconds before resuming detection.")

    def capture_imgs(self, img: np.ndarray) -> None:
        """Captures and saves the image."""
        print(f"Capturing image {self.delay_number + 1}")
        filename = f"{int(self.first_trigger_time)}_{self.delay_number}.jpg"  # Use first_trigger_time to name the files
        filepath = os.path.join(self.folder_name, filename)
        cv2.imwrite(filepath, img)
        self.delay_number += 1

    def wait_for_delay(self, start_time: float, delay_time: float) -> bool:
        """Waits for the specified delay time before continuing."""
        return time.time() - start_time >= delay_time

    
    def save_img(self, img: np.ndarray, num_of_img: int) -> None:
        filename = f"{self.first_trigger_time}_{num_of_img}.jpg"
        filepath = os.path.join(self.folder_name, filename)
        cv2.imwrite(filepath, img)

class CustomCNN(nn.Module):
    def __init__(self, layers: List[int], img_height: int = 150):
        super(CustomCNN, self).__init__()
        self.layers = nn.ModuleList()
        input_channels = 3

        for output_channels in layers:
            self.layers.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1))
            self.layers.append(nn.BatchNorm2d(output_channels))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.MaxPool2d(2))
            input_channels = output_channels

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_channels * (img_height // 2**len(layers))**2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.classifier(x)
        return x

class PhoneDetector:
    def __init__(self, model_path: str, confidence_threshold: float = 0.53, img_height: int = 150, img_width: int = 150):
        self.img_height = img_height
        self.img_width = img_width
        self.confidence_threshold = confidence_threshold
        
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
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = self.transform(image)
        image = image.unsqueeze(0)
        return image

    def detect_phone(self, frame: np.ndarray) -> (bool, float):
        processed_image = self.preprocess_image(frame)
        with torch.no_grad():
            outputs = self.model(processed_image)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        print(f'Predicted: {predicted.item()}, Confidence: {confidence.item()}')
        return predicted.item() == 1, confidence.item()

