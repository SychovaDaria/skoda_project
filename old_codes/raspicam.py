"""
Module for providing camera functionality on a Raspberry Pi either using the camera module,
or a USB camera.

Author: Josef Kahoun
Date: 17.06.2024
Version: 0.1
Comments: v4l2-ctl --list-devices to list USB cameras, select the first one under the camera as the usb_camera_index
"""

import cv2
import numpy as np
import os

from picamera2 import Picamera2
from typing import List, Tuple

# Constants for default values
DEFAULT_RESOLUTION = (2028, 1520)
DEFAULT_EXPOSURE_VALUE = 0.0
DEFAULT_SATURATION = 1.0
DEFAULT_SHARPNESS = 1.0



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
    
    Example:
        cam = Raspicam()
        cam.set_controls(exposure_value=4.0, saturation=16.0, sharpness=8.0)
        img = cam.capture_img() # returns a numpy array that can be used for further processing
        cam.capture_img_and_save("image.jpg") # saves the image to the current directory
        cam.stop()
    """
    def __init__(self, resolution: Tuple[int,int] = DEFAULT_RESOLUTION, exposure_value: float = DEFAULT_EXPOSURE_VALUE,
                 saturation: float =DEFAULT_SATURATION, sharpness: float = DEFAULT_SHARPNESS, use_usb: bool = False) -> None: 
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
            # TODO: add camera settings
            self.camera = cv2.VideoCapture()

    def set_controls(self, exposure_value: float = DEFAULT_EXPOSURE_VALUE, saturation: float = DEFAULT_SATURATION,
                     sharpness: float = DEFAULT_SHARPNESS) -> None:
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
        else:
            pass
            #TODO: add settings for USB camera

    def set_default_controls(self) -> None:
        """
        Sets the camera controls to default

        Args:
            None
        
        Returns:
            None
        """
        self.set_controls(exposure_value=DEFAULT_EXPOSURE_VALUE,saturation=DEFAULT_SATURATION,sharpness=DEFAULT_SHARPNESS)

    def start(self) -> None:
        """
        Starts the camera, it is automatically called when creating raspicam object.

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
            img (np.array) - 3D numpy array containing the current img (row, height, RGB)
        """
        if self.use_usb:
            ret, image = self.camera.read()
            if ret is False:
                raise KeyError("Failed to read image from USB camera")
        else:
            image = self.camera.capture_array()
            image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        return image

    def capture_img_and_save(self, filename: str, folder_path: str = "") -> None:
        """
        Saves the current img to the desired folder (if folder doesn't exist, it creates it)
        using the provided filename.

        Args:
            folder_path (str): The path of the folder where the image will be saved.
            filename (str): The name of the image file.

        Returns:
            None
        """
        if not filename.endswith((".jpg", ".png")):
            raise ValueError("Invalid filename. Filename must end with .jpg or .png")
        image = self.capture_img()
        if not os.path.exists(folder_path) and folder_path != "":
            os.makedirs(folder_path)
        if folder_path == "":
            cv2.imwrite(filename,image)
        else:
            cv2.imwrite(folder_path+"/"+filename,image)