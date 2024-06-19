"""
Module for providing camera functionality on a Raspberry Pi either using the camera module,
or a USB camera.

Author: Josef Kahoun
Date: 18.06.2024
Version: 0.2
Comments: 
    v4l2-ctl --list-devices to list USB cameras, select the first one under the camera as the usb_camera_index

Improvements:
    Add a functionality, that will write default values based on the connected sensor to a file at the start of the
    program. This will ensure functionality across all sensors.
"""

import cv2
import numpy as np
import os

from picamera2 import Picamera2
from typing import List, Tuple

# Constants for default values
DEFAULT_RESOLUTION = (2028, 1520)
DEFAULT_EXPOSURE_VALUE = 0 # range -8.0 to 8.0 # FIXME: doesnt do anything, delete
DEFAULT_SATURATION = 1 # range 0 to 32.0
DEFAULT_SHARPNESS = 1 # range 0 to 16.0
DEFAULT_FRAMERATE = 30
DEFAULT_BRIGHTNESS = 0 # range -1.0 to 1.0
DEFAULT_CONTRAST = 1 # range 0.0 to 32.0

USB_DEFAULT_RESOLUTION = (640, 480)
USB_DEFAULT_SATURATION = 100 # range 0 - 200
USB_DEFAULT_SHARPNESS = 25 # range 0 - 50
USB_DEFAULT_FRAMERATE = 30
USB_DEFAULT_CONTRAST = 5.0 # range 0 - 10.0
USB_DEFAULT_BRIGHTNESS = 133.0 # range 30.0 - 255.0



class Raspicam:
    """
    Object for handling the camera on a Raspberry Pi.

    Args:
        resolution (Tuple[int, int], optional): The resolution of the camera. Defaults to None.
        framerate (int, optional): The framerate of the camera. Defaults to None.
        exposure_value (float, optional): The exposure value of the camera.
        saturation (float, optional): The saturation value of the camera.
        sharpness (float, optional): The sharpness value of the camera.
        use_usb (bool, optional): Whether to use a USB camera. Defaults to False.
        brightness (float, optional): The brightness value of the camera.
        contrast (float, optional): The contrast value of the camera.

    Attributes:
        resolution (Tuple[int, int]): The resolution of the camera.
        exposure_value (float): The exposure time of the camera.
        framerate (int): The framerate of the camera.
        saturation (float): The saturation value of the camera.
        sharpness (float): The sharpness value of the camera.
        use_usb (bool): Whether to use a USB camera.
        brightness (float): The brightness value of the camera.
        contrast (float): The contrast value of the camera.
        camera (Union[Picamera2, cv2.VideoCapture]): The camera object.
    
    Example:
        cam = Raspicam()
        cam.set_controls(exposure_value=4.0, saturation=16.0, sharpness=8.0)
        img = cam.capture_img() # returns a numpy array that can be used for further processing
        cam.capture_img_and_save("image.jpg") # saves the image to the current directory
        cam.stop()
    """
    def __init__(self, resolution: Tuple[int,int] = None, exposure_value: float = None,
                 saturation: float =None, sharpness: float = None, 
                 framerate : int = None, use_usb: bool = False, brightness: float = None,
                 contrast: float = None) -> None: 
        self.use_usb = use_usb
        if use_usb: # create defaults
            self.resolution = USB_DEFAULT_RESOLUTION
            self.resolution = USB_DEFAULT_RESOLUTION
            self.saturation = USB_DEFAULT_SATURATION
            self.exposure_value = DEFAULT_EXPOSURE_VALUE
            self.sharpness = USB_DEFAULT_SHARPNESS
            self.framerate = USB_DEFAULT_FRAMERATE
            self.brightness = USB_DEFAULT_BRIGHTNESS
            self.contrast = USB_DEFAULT_CONTRAST
        else:
            self.resolution = DEFAULT_RESOLUTION
            self.exposure_value = DEFAULT_EXPOSURE_VALUE
            self.saturation = DEFAULT_SATURATION
            self.sharpness = DEFAULT_SHARPNESS
            self.framerate = DEFAULT_FRAMERATE
            self.brightness = DEFAULT_BRIGHTNESS
            self.contrast = DEFAULT_CONTRAST
        if resolution is not None:
            self.resolution = resolution
        # start the cam and set desired controls
        if not use_usb: # start picam if not using usb
            self.camera = Picamera2()
            camera_config = self.camera.create_preview_configuration(main={'size': self.resolution})
            self.camera.configure(camera_config)
            self.set_controls(exposure_value=exposure_value,saturation=saturation,sharpness=sharpness,
                          framerate=framerate,brightness=brightness, contrast=contrast)
            self.camera.start()
        else:
            self.camera = cv2.VideoCapture(8) # TODO: automatically learn index of the camera
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.set_controls(exposure_value=exposure_value,saturation=saturation,sharpness=sharpness,
                          framerate=framerate,brightness=brightness, contrast=contrast)

    def set_controls(self, exposure_value: float = None, saturation: float = None,
                     sharpness: float = None, framerate: int = None, brightness: float = None,
                     contrast : float = None) -> None:
        """
        Sets the camera controls.

        Args:
            exposure_value (float, optional): The exposure value to set.
            saturation (float, optional): The saturation value to set.
            sharpness (float, optional): The sharpness value to set.
            framerate (int, optional): The framerate value to set.
            brightness (float, optional): The brightness value to set.
            contrast (float, optional): The contrast value to set.

        Returns:
            None
        """
        if exposure_value is not None:
            self.exposure_value = exposure_value
        if saturation is not None:
            self.saturation = saturation
        if sharpness is not None:
            self.sharpness = sharpness
        if framerate is not None:
            self.framerate = framerate
        if brightness is not None:
            self.brightness = brightness
        if contrast is not None:
            self.contrast = contrast

        if not self.use_usb:
            self.camera.set_controls({"ExposureValue": self.exposure_value, "Saturation": self.saturation, 
                                      "Sharpness": self.sharpness, "FrameRate": self.framerate,
                                      "Brightness": self.brightness,"Contrast": self.contrast})
        else:
            self.camera.set(cv2.CAP_PROP_SATURATION, self.saturation)
            self.camera.set(cv2.CAP_PROP_SHARPNESS, self.sharpness)
            self.camera.set(cv2.CAP_PROP_FPS, self.framerate)
            self.camera.set(cv2.CAP_PROP_BRIGHTNESS, self.brightness)
            self.camera.set(cv2.CAP_PROP_CONTRAST, self.contrast)

    def set_default_controls(self):
        """
        Sets the default controls for the camera.

        If `use_usb` is True, the controls for a USB camera are set to the default values.
        If `use_usb` is False, the controls for a default camera are set to the default values.
        
        Returns:
            None
        """
        if self.use_usb:
            self.change_resolution(resolution=USB_DEFAULT_RESOLUTION)
            self.set_controls(exposure_value=DEFAULT_EXPOSURE_VALUE, saturation=USB_DEFAULT_SATURATION, 
                              sharpness=USB_DEFAULT_SHARPNESS,framerate=USB_DEFAULT_FRAMERATE,
                              brightness=USB_DEFAULT_BRIGHTNESS, contrast=USB_DEFAULT_CONTRAST)
        else:
            self.change_resolution(resolution=DEFAULT_RESOLUTION)
            self.set_controls(exposure_value=DEFAULT_EXPOSURE_VALUE, saturation=DEFAULT_SATURATION, 
                              sharpness=DEFAULT_SHARPNESS,framerate=DEFAULT_FRAMERATE,
                              brightness=DEFAULT_BRIGHTNESS, contrast=DEFAULT_CONTRAST)


    def change_resolution(self, resolution: Tuple[int, int]) -> None:
        """
        Changes the resolution of the camera.

        Args:
            resolution (Tuple[int, int]): The resolution to set.

        Returns:
            None
        """
        self.resolution = resolution
        if not self.use_usb:
            self.stop() # the camera needs to be stopped to change the resolution
            camera_config = self.camera.create_preview_configuration(main={'size': resolution})
            self.camera.configure(camera_config)
            self.start()
        else:
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

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
        print(f"\tFramerate: {self.framerate}")
        print(f"\tBrightness: {self.brightness}")
        print(f"\tContrast: {self.contrast}")


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

    def change_camera_feed(self):
        """
        Changes the camera feed from the picamera to the usb camera and vice versa.

        Returns:
            None 
        """
        self.stop()
        self.__init__(use_usb=not self.use_usb)

    def auto_brightness(self, target_brightness: float, brightness_threshold : float = 0.2,
                        max_iterations: int = 10): # FIXME: doesnt work
        """
        Adjusts the exposure value to achieve the target brightness in the image.

        Args:
            target_brightness (float): The desired target brightness value.
            max_iterations (int, optional): The maximum number of iterations to adjust the exposure value. Defaults to 10.

        Returns:
            None
        """
        if self.use_usb:
            raise ValueError("Auto brightness adjustment is only supported for the PiCamera.")

        # Initialize the exposure value
        exposure_value = self.exposure_value

        # Iterate to adjust the exposure value
        for i in range(max_iterations):
            # Capture an image
            image = self.capture_img()

            # Calculate the current brightness
            current_brightness = self.calculate_brightness(image)

            # Check if the current brightness is close to the target brightness
            if abs(current_brightness - target_brightness) < brightness_threshold:
                break

            # Adjust the exposure value based on the difference between the current and target brightness
            exposure_value += (target_brightness - current_brightness) * brightness_threshold

            # Set the new exposure value
            print("DONE")
            self.set_controls(exposure_value=max(min(-8.0,exposure_value),8.0))

        # Print the final settings
        self.print_settings()

    def calculate_brightness(self, image: np.array = None) -> float:
        """
        Calculates the brightness of the image.

        Args:
            image (np.array, optional): The image to calculate the brightness. If not provided, the current image will be used.

        Returns:
            float: The brightness value.
        """
        if image is None:
            image = self.capture_img()

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate the mean brightness
        mean_brightness = np.mean(gray_image)

        return mean_brightness