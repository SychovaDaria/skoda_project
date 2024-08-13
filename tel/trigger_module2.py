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
            nn.Linear(input_channels * (img_height // 2**len(layers))**2, 256),  # First classifier layer
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),  # Second classifier layer
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)  # Using 2 outputs for binary classification
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.classifier(x)
        return x

class PhoneDetector:
    def __init__(self, model_path, camera, img_height=150, img_width=150, capture_interval=20, update_ui_callback=None):
        self.img_height = img_height
        self.img_width = img_width
        self.capture_interval = capture_interval
        self.last_detection_time = 0
        
        self.camera = camera  #using inicialized camera
        self.update_ui_callback = update_ui_callback
        
        # Загрузка модели
        self.model = CustomCNN([64, 128])
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        os.makedirs('object', exist_ok=True)
        self.running = True

    def preprocess_image(self, image):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = self.transform(image)
        image = image.unsqueeze(0)
        return image

    def detect_phone(self, frame):
        processed_image = self.preprocess_image(frame)
        with torch.no_grad():
            outputs = self.model(processed_image)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        return predicted.item() == 1 and confidence.item() >= 0.95

    def capture_images(self, frame):
        for i in range(3):
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join('object', f'object_detected_{timestamp}.jpg')
            cv2.imwrite(filename, frame)
            time.sleep(0.5)

    def run(self):
        while self.running:
            current_time = time.time()
            frame = self.camera.capture_img()

            frame = cv2.flip(frame, -1)

            if current_time - self.last_detection_time > self.capture_interval:
                if self.detect_phone(frame):
                    capture_thread = Thread(target=self.capture_images, args=(frame,))
                    capture_thread.start()
                    self.last_detection_time = time.time()
                else:
                    print("No object detected.")

            if self.update_ui_callback:
                self.update_ui_callback(frame)

            if cv2.waitKey(1) == 27:
                break

        cv2.destroyAllWindows()

    def stop(self):
        self.running = False

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1]
    camera = Raspicam(use_usb=False)  # inicialize camera
    detector = PhoneDetector(model_path=model_path, camera=camera, img_height=150, img_width=150, capture_interval=20)
    detector.run()

