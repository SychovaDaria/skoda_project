"""
Module for providing camera functionality on a Raspberry Pi either using the camera module,
 or a USB camera.

Author: Josef Kahoun
Date: 17.06.2024
"""

import cv2
import numpy as np
import os

from picamera2 import Picamera2
from typing import List, Tuple


class Raspicam:
    """
    Object for handling the camera on a Raspberry Pi.

    Args:
        resolution (Tuple[int, int]): The resolution of the camera.
        framerate (int): The framerate of the camera.
        exposure_value (int): The exposure value of camera, ranging -8.0 to 8.0, more being brighter
        saturation (float): The saturation value of the camera, Floating point number from 0.0 to 32.0
        sharpness (float): The sharpness value of the camera, Floating point number from 0.0 to 16.0
        use_usb (bool, optional): Whether to use a USB camera. Defaults to False.

    Attributes:
        resolution (Tuple[int, int]): The resolution of the camera.
        exposure_value (int): The exposure time of the camera.
        framerate (int): The framerate of the camera.
        saturation (float): The saturation value of the camera.
        sharpness (float): The sharpness value of the camera.
        use_usb (bool): Whether to use a USB camera.
        camera (Union[Picamera2, cv2.VideoCapture]): The camera object.
    """
    # default resolution --> 2028x1520
    def __init__(self, resolution: Tuple[int,int], exposure_value: float = 0.0,
                 saturation: float = 1.0, sharpness: float = 1.0, use_usb: bool = False) -> None: 
        self.resolution = resolution
        self.exposure_value = exposure_value
        self.saturation = saturation
        self.sharpness = sharpness
        self.use_usb = use_usb
        if not use_usb: # start picam if not using usb
            self.camera = Picamera2()
            camera_config = self.camera.create_preview_configuration(main={'size': resolution})
            self.camera.configure(camera_config)
            self.camera.set_controls({"ExposureValue": exposure_value, "Saturation": saturation, 
                                      "Sharpness": sharpness})
            self.camera.start()
        else:
            # FIXME: add camera settings
            self.camera = cv2.VideoCapture(0)

    def set_controls(self, exposure_value: int = 1000, saturation: float = 16.0,
                     sharpness: float = 8.0) -> None:
        """
        Sets the camera controls.

        Args:
            exposure_value (int): The exposure time to set.
            saturation (float): The saturation value to set.
            sharpness (float): The sharpness value to set.

        Returns:
            None
        """
        self.exposure_value = exposure_value
        self.saturation = saturation
        self.sharpness = sharpness
        if not self.use_usb:
            self.camera.set_controls({"ExposureTime": exposure_value, "Saturation": saturation, 
                                      "Sharpness": sharpness})

    def start(self) -> None:
        """
        Starts the camera.

        Returns:
            None
        """
        if not self.use_usb:
            self.camera.start()
    
    def stop(self) -> None:
        """
        Stops the camera.

        Returns:
            None
        """
        if not self.use_usb:
            self.camera.stop()

    def print_settings(self) -> None:
        """
        Prints the current camera settings.

        Returns:
            None
        """
        print("Current camera settings:")
        print(f"\tResolution: {self.resolution}")
        print(f"\tExposure time: {self.exposure_value}")
        print(f"\tSaturation: {self.saturation}")
        print(f"\tSharpness: {self.sharpness}")

    def capture_img(self) -> np.array:
        """
        Captures the current img
        
        Returns:
            img (np.array) - 3D numpy array containing the current img
        """
        if self.use_usb:
            ret, image = self.camera.read()
            if ret is False:
                raise KeyError("Failed to read image from USB camera")
        else:
            image = self.camera.capture_array()
        return image

    def capture_img_and_save(self, filename: str, folder_path: str = "") -> None:
        """
        Saves the current img to the desired folder (if folder doesnt exist, it creates it)
        using the provided filename.

        Args:
            folder_path (str): The path of the folder where the image will be saved.
            filename (str): The name of the image file.

        Returns:
            None
        """
        if not os.path.exists(folder_path) and folder_path != "":
            os.makedirs(folder_path)
        if folder_path == "":
            self.camera.capture_file(filename)
        else:
            self.camera.capture_file(folder_path+"/"+filename)