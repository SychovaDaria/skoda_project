"""
Module for picture aquisition from the camera.

This module gets a picture and the boolean value of the object detection in the picture. The picture is saved in 
the folder with the name of the current date and time. The module contains a class for handling the 
aquisition of pictures from the camera.

Author: Josef Kahoun
Date: 14. 8. 2024
Version: 0.2
"""
import cv2
from enum import Enum
import numpy as np
from raspicam import Raspicam
from typing import List
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import time

# default values
DEFAULT_TRIGGER_DELAY = 0
DEFAULT_NUM_OF_PICTURES = 1
DEFAULT_TIMES_BETWEEN_PICTURES = .5

class TriggerModes(Enum):
    """
    Enum for the trigger modes.

    Attributes:
        ALWAYS (int): The trigger is always on (only off because of the time delay between captures).
        RISING_EDGE (int): The trigger is on only when the trigger signal goes from low to high.       
        FALLING_EDGE (int): The trigger is on only when the trigger signal goes from high to low.
    """
    ALWAYS = 0
    RISING_EDGE = 1
    FALLING_EDGE = 2


class Trigger:
    """
    Object for handling aquisition of pictures from the camera.

    Args:
        folder_name (str): The folder where the pictures will be saved.
        trigger_delay (float, optional): The delay between the trigger and the start of the acquisition. Defaults to DEFAULT_TRIGGER_DELAY.
        num_of_pictures (int, optional): The number of pictures to take. Defaults to DEFAULT_NUM_OF_PICTURES.
        times_between_pictures (float|List[float], optional): The time between pictures. Can be a single float or a list of floats. If it is a float, the times between the pictures are 
        the same. Defaults to DEFAULT_TIMES_BETWEEN_PICTURES.
    
    Attributes:
        trigger_delay (float): The delay between the trigger and the start of the acquisition.
        num_of_pictures (int): The number of pictures to take.
        times_between_pictures (float|List[float]): The time between pictures. Can be a single float or a list of floats.
        folder_name (str): The folder where the pictures will be saved.
    """
    def __init__(self, folder_name : str, trigger_delay : float = DEFAULT_TRIGGER_DELAY,
                 num_of_pictures : int = 1, time_to_reset : float = 0,
                 times_between_pictures : float|List[float] = DEFAULT_TIMES_BETWEEN_PICTURES,
                 trigger_mode: TriggerModes = TriggerModes.ALWAYS) -> None:
        self.trigger_delay = trigger_delay
        self.time_to_reset = time_to_reset
        self.num_of_pictures = num_of_pictures
        self.times_between_pictures = times_between_pictures
        if isinstance(self.times_between_pictures, float|int):
                self.times_between_pictures = [self.times_between_pictures] * (self.num_of_pictures - 1)
        self.folder_name = folder_name
        self.trigger_mode = trigger_mode
        self.last_trigger_signal = False
        self.last_trigger_time = time.time()
        self.first_trigger_time = time.time()
        self.capturing = False
        self.delay_number = 0
        self.file_name = ""
        self.check_attributes()
    
    def check_attributes(self) -> None:
        """
        Checks if the attributes are valid.

        Returns:
            None
        """
        if not isinstance(self.trigger_delay, float|int) or self.trigger_delay < 0:
            raise ValueError("The trigger_delay attribute must be a non-negative float.")
        if not isinstance(self.time_to_reset, float|int) or self.time_to_reset < 0:
            raise ValueError("The time_to_reset attribute must be a non-negative float.")
        if not isinstance(self.num_of_pictures, int) or self.num_of_pictures < 1:
            raise ValueError("The num_of_pictures attribute must be a positive integer.")
        if isinstance(self.times_between_pictures, float|int):
            if self.times_between_pictures < 0:
                raise ValueError("The times_between_pictures attribute must be a non-negative float or a list of non-negative floats.")
        elif isinstance(self.times_between_pictures, list):
            if any(not isinstance(time, float|int) or time < 0 for time in self.times_between_pictures):
                raise ValueError("The times_between_pictures attribute must be a non-negative float or a list of non-negative floats.")
            if len(self.times_between_pictures) != self.num_of_pictures - 1:
                raise ValueError("The length of the times_between_pictures list must be equal to num_of_pictures - 1., or times_between_pictures must be a float.")
        else:
            raise ValueError("The times_between_pictures attribute must be a non-negative float or a list of non-negative floats.")
        if self.folder_name is not None and (not (isinstance(self.folder_name, str) and os.path.isdir(self.folder_name))):
            raise ValueError("The folder_name attribute must be a string representing a valid directory path.")
        if not isinstance(self.trigger_mode, TriggerModes):
            raise ValueError("The trigger_mode attribute must be an instance of the TriggerModes enum.")
        if self.file_name is not None and not(isinstance(self.file_name, str)):
            raise ValueError("The file_name attribute must be a string.")

    def set_config(self, folder_name : str = None, trigger_delay : float = None, time_to_reset : float = None,
                   num_of_pictures: int = None, times_between_pictures : float|List[float] = None,
                   trigger_mode: TriggerModes = None, file_name: str = None) -> None:
        """
        Set the configuration of the trigger.

        Args:
            folder_name (str, optional): The folder where the pictures will be saved.
            trigger_delay (float, optional): The delay between the trigger and the start of the acquisition.
            num_of_pictures (int, optional): The number of pictures to take.
            times_between_pictures (float|List[float], optional): The time between pictures. Can be a single float or a list of floats.

        Returns:
            None
        """
        if folder_name is not None:
            self.folder_name = folder_name
        if trigger_delay is not None:
            self.trigger_delay = trigger_delay
        if time_to_reset is not None:
            self.time_to_reset = time_to_reset
        if num_of_pictures is not None:
            self.num_of_pictures = num_of_pictures
        if times_between_pictures is not None:
            self.times_between_pictures = times_between_pictures
            # transform times_between_pictures to a list if it is a float
            if isinstance(self.times_between_pictures, float|int):
                self.times_between_pictures = [self.times_between_pictures] * (self.num_of_pictures - 1)
        if trigger_mode is not None:
            self.trigger_mode = trigger_mode
        if file_name is not None:
            self.file_name = file_name
        self.check_attributes()


    def process_trigger_signal(self, trigger_signal: bool, img : np.ndarray) -> None:
        """
        Process the trigger signal.

        Args:
            trigger_signal (bool): The trigger signal. If True, the object in the image was detected, if False, the object was not detected
            img (np.ndarray): The image to save.

        Returns:
            None
        """
        if (not self.capturing and 
            (
            ((self.trigger_mode == TriggerModes.ALWAYS) and trigger_signal) or 
            ((self.trigger_mode == TriggerModes.RISING_EDGE) and trigger_signal and not self.last_trigger_signal) or 
            ((self.trigger_mode == TriggerModes.FALLING_EDGE) and not trigger_signal and self.last_trigger_signal)
            )
        ):
            print("Started capturing")
            self.capturing = True
            self.last_trigger_time = time.time()
            self.first_trigger_time = self.last_trigger_time
        self.last_trigger_signal = trigger_signal
        if self.capturing:
            self.capture_imgs(img)

    def capture_imgs(self, img:np.ndarray) -> None:
        """
        Function for capturing the images.
        Implements the delays non-blocking way.

        Args:
            img (np.ndarray): The current frame from the camera stream.

        Returns:
            None
        """
        # wait for the initial delay
        if self.delay_number == 0:
            print("waiting for the initial delay")
            if not self.wait_for_delay(self.last_trigger_time, self.trigger_delay):
                return
            self.last_trigger_time = time.time()
            self.save_img(img,self.delay_number)
            self.delay_number += 1
        # the delays between the pictures
        if self.delay_number < self.num_of_pictures:
            print(f"waiting for the delay number {self.delay_number}")
            wait_time = self.times_between_pictures[self.delay_number - 1]
            if not self.wait_for_delay(self.last_trigger_time, wait_time):
                return
            self.last_trigger_time = time.time()
            self.save_img(img,self.delay_number)
            self.delay_number += 1
        else: 
            # wait for the reset delay, the start capturing again
            if not self.wait_for_delay(self.first_trigger_time, self.time_to_reset):
                return
            self.capturing = False
            self.delay_number = 0
    
    def wait_for_delay(self, start_time: float, delay_time: float) -> bool:
        """
        Returns True if the delay has passed, False otherwise.

        Args:
            start_time (float): The time when the delay started.
            delay_time (float): The time of the delay.
        
        Returns:
            bool: True if the delay has passed, False otherwise.
        """
        return time.time() - start_time >= delay_time
    
    def save_img(self, img:np.ndarray, num_of_img: int) -> None:
        """
        Save the image to the folder.

        Args:
            img (np.ndarray): The image to save.

        Returns:
            None
        """
        if self.file_name == "":
            filename = f"{self.first_trigger_time}_{num_of_img}.jpg"
        else:
            filename = f"{self.file_name}_{self.first_trigger_time}_{num_of_img}.jpg"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(self.folder_name+"/"+filename,img)

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
    def __init__(self, model_path: str, confidence_threshold: float = 0.85, img_height: int = 150, img_width: int = 150):
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
        return (predicted.item() == 1) and (confidence.item()>self.confidence_threshold)