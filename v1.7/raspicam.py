"""
Module for providing camera functionality on a Raspberry Pi either using the camera module,
or a USB camera.

Author: Josef Kahoun
Date: 19.06.2024
Version: 0.3
Comments: 
    v4l2-ctl --list-devices to list USB cameras, select the first one under the camera as the usb_camera_index

Improvements:
    Add a functionality, that will write default values based on the connected sensor to a file at the start of the
    program. This will ensure functionality across all sensors.
    Make it so you don't have to manually write USB camera index.
    Add error handling and arguments checker.
"""

import cv2
import numpy as np
import os
import sys
from typing import List, Tuple

from picamera2 import Picamera2


# Constants for default values
DEFAULT_RESOLUTION = (2028, 1520)
DEFAULT_EXPOSURE_VALUE = 0 # range -8.0 to 8.0
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

DEFAULT_AUTO_BRIGHTNESS_VALUE = 133
AUTO_BRIGHTNESS_ERROR_THRESHOLD = 3

class Raspicam:
    """
    Object for handling the camera on a Raspberry Pi.

    Object for handling the camera on a Raspberry Pi. It is designed to drive both pi camera module connected 
    with ribbon cable, or an external USB camera. The default settings were set for the 13MP camera module and 
    Microsoft LifeCam Studio camera.

    Args:
        resolution (Tuple[int, int], optional): The resolution of the camera. Defaults to None.
        framerate (int, optional): The framerate of the camera. Defaults to None.
        exposure_value (float, optional): The exposure value of the camera.
        saturation (float|int, optional): The saturation value of the camera.
        sharpness (float, optional): The sharpness value of the camera.
        use_usb (bool, optional): Whether to use a USB camera. Defaults to False.
        brightness (float, optional): The brightness value of the camera.
        contrast (float, optional): The contrast value of the camera.
        auto_exposure_on (bool, optional): Whether to turn on auto exposure. Defaults to False.
        auto_brightness_value (float, optional): The target brightness value for auto brightness adjustment. Defaults to DEFAULT_AUTO_BRIGHTNESS_VALUE.

    Attributes:
        resolution (Tuple[int, int]): The resolution of the camera.
        exposure_value (float): The exposure time of the camera.
        framerate (int): The framerate of the camera.
        saturation (float|int): The saturation value of the camera.
        sharpness (float): The sharpness value of the camera.
        use_usb (bool): Whether to use a USB camera.
        brightness (float): The brightness value of the camera.
        contrast (float): The contrast value of the camera.
        auto_exposure_on (bool): Whether auto exposure is turned on.
        auto_brightness_value (float): The target brightness value for auto brightness adjustment.
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
                 contrast: float = None, auto_exposure_on: bool = False, 
                 auto_brightness_value : float = DEFAULT_AUTO_BRIGHTNESS_VALUE) -> None: 
        self.use_usb = use_usb
        if self.use_usb:
            self.resolution = USB_DEFAULT_RESOLUTION
            self.saturation = USB_DEFAULT_SATURATION
            self.exposure_value = DEFAULT_EXPOSURE_VALUE
            self.sharpness = USB_DEFAULT_SHARPNESS
            self.framerate = USB_DEFAULT_FRAMERATE
            self.brightness = USB_DEFAULT_BRIGHTNESS
            self.contrast = USB_DEFAULT_CONTRAST
        else:
            self.resolution = DEFAULT_RESOLUTION
            self.saturation = DEFAULT_SATURATION
            self.exposure_value = DEFAULT_EXPOSURE_VALUE
            self.sharpness = DEFAULT_SHARPNESS
            self.framerate = DEFAULT_FRAMERATE
            self.brightness = DEFAULT_BRIGHTNESS
            self.contrast = DEFAULT_CONTRAST

        self.turn_auto_exposure_on = auto_exposure_on
        self.auto_brightness_value = auto_brightness_value
        if resolution is not None:
            self.resolution = resolution
        self.check_attributes()
        # start the cam and set desired controls
        if not use_usb: # start picam if not using usb
            self.camera = Picamera2()
            camera_config = self.camera.create_preview_configuration(main={'size': self.resolution})
            self.camera.configure(camera_config)
            self.set_controls(exposure_value=exposure_value,saturation=saturation,sharpness=sharpness,
                          framerate=framerate,brightness=brightness, contrast=contrast)
            self.camera.start()
        else:
            if sys.platform == "win32":
                self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW) # TODO: automatically learn index of the camera
            else:
                self.camera = cv2.VideoCapture(0, cv2.CAP_ANY)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.set_controls(exposure_value=exposure_value,saturation=saturation,sharpness=sharpness,
                          framerate=framerate,brightness=brightness, contrast=contrast)

    def set_controls(self, exposure_value: float = None, saturation: float|int = None,
                     sharpness: float|int = None, framerate: int = None, brightness: float = None,
                     contrast : float = None, auto_exposure_on: bool = None, 
                     auto_brightness_value : float = None) -> None:
        """
        Sets the camera controls.

        Args:
            exposure_value (float, optional): The exposure value to set.
            saturation (float, optional): The saturation value to set.
            sharpness (float, optional): The sharpness value to set.
            framerate (int, optional): The framerate value to set.
            brightness (float, optional): The brightness value to set.
            contrast (float, optional): The contrast value to set.
            auto_exposure_on (bool, optional): Whether to turn on auto exposure.
            auto_brightness_value (float, optional): The target brightness value for auto brightness adjustment.

        Returns:
            None
        """
        if auto_exposure_on is not None:
            self.turn_auto_exposure_on = auto_exposure_on
        if exposure_value is not None and not self.turn_auto_exposure_on:
            self.exposure_value = exposure_value
        if saturation is not None:
            self.saturation = saturation
            if not self.use_usb:
                self.saturation = self.saturation * 32/200
        if sharpness is not None:
            self.sharpness = sharpness
            if not self.use_usb:
                self.sharpness = self.sharpness * 16/50
        if framerate is not None:
            self.framerate = framerate
        if brightness is not None:
            self.brightness = brightness
            if not self.use_usb:
                self.brightness = ((self.brightness-30)*2/255)-1
        if contrast is not None:
            self.contrast = contrast
            if not self.use_usb:
                self.contrast = self.contrast * 32/10
        #FIXME: not tested the conversion, rather check when testing on raspberry
        if auto_brightness_value is not None:
            self.auto_brightness_value = auto_brightness_value
        self.check_attributes()
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
                              brightness=USB_DEFAULT_BRIGHTNESS, contrast=USB_DEFAULT_CONTRAST, auto_exposure_on=False,
                              auto_brightness_value=DEFAULT_AUTO_BRIGHTNESS_VALUE)
        else:
            self.change_resolution(resolution=DEFAULT_RESOLUTION)
            self.set_controls(exposure_value=DEFAULT_EXPOSURE_VALUE, saturation=DEFAULT_SATURATION, 
                              sharpness=DEFAULT_SHARPNESS,framerate=DEFAULT_FRAMERATE,
                              brightness=DEFAULT_BRIGHTNESS, contrast=DEFAULT_CONTRAST, auto_exposure_on=False,
                              auto_brightness_value=DEFAULT_AUTO_BRIGHTNESS_VALUE)


    def change_resolution(self, resolution: Tuple[int, int]) -> None:
        """
        Changes the resolution of the camera.

        Args:
            resolution (Tuple[int, int]): The resolution to set.

        Returns:
            None
        """
        self.resolution = resolution
        self.check_attributes()
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
        print(f"\tAuto Exposure On: {self.turn_auto_exposure_on}")
        print(f"\tAuto Brightness Value: {self.auto_brightness_value}")


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
            
            image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        else:
            image = self.camera.capture_array()
            image = np.flip(image, axis = (0,1))
            #image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            if self.turn_auto_exposure_on: # only turn auto brightness on picamera2, not when using USB
                self.auto_brightness(image)
        return image

    def capture_img_and_save(self, filename: str, folder_path: str = "") -> None:
        """
        Saves the current img to the desired folder. (if folder doesn't exist, it creates it)
        using the provided filename.

        Saves the current img to the desired folder. If the folder doesn't exist, it creates it
        using the provided folder_path name. Can save pictures in .jpg and in .png format.
        Args:
            filename (str): The name of the image file.
            folder_path (str, optional): The path of the folder where the image will be saved.

        Returns:
            None
        """
        if not filename.endswith((".jpg", ".png")):
            raise ValueError("Invalid filename. Filename must end with .jpg or .png")
        image = self.capture_img()
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        if not os.path.exists(folder_path) and folder_path != "":
            os.makedirs(folder_path)
        if folder_path == "":
            cv2.imwrite(filename,image)
        else:
            cv2.imwrite(folder_path+"/"+filename,image)

    def change_camera_feed(self) -> None:
        """
        Changes the camera feed from the picamera to the usb camera and vice versa.

        Changes the camera feed from the picamera to the usb camera and vice versa. Sets all the settings to default,
        so manual change of the settings is needed if desired.

        Returns:
            None 
        """
        self.stop()
        self.__init__(use_usb=not self.use_usb)
        
    # INFO: the picamera has some auto exposure settings, so probably not needed, but does offer more customization
    # regarding setting up the parameter, which the picamera does not have
    def auto_brightness(self, image: np.array = None, roi : List[int] = None) -> None:
        """
        Automatically adjusts the brightness of the image.
        
        Automatically adjusts the brightness of the image by changing exposure value
        using P regulator. Need to call the capture_img function in a loop in order for this to work.

        Args:
            image (np.array, optional): The image to adjust the brightness. If not provided, the current image will be used.
            roi (List[int], optional): The region of interest to calculate the brightness. If not provided, the whole image will be used. x1, y1, x2, y2

        Returns:
            None
        """
        if image is None:
            image = self.capture_img()
        if roi is not None:
            image = image[roi[0]:roi[2], roi[1]:roi[3]]
        brightness = self.calculate_brightness(image)
        K_p = 0.0005 # the P regulator constant
        error = self.auto_brightness_value - brightness
        correction = K_p * error
        if abs(error) < AUTO_BRIGHTNESS_ERROR_THRESHOLD:
            return
        # cap the exposure_value
        if self.exposure_value + correction < -8.0:
            self.set_controls(exposure_value=-8.0)
        elif self.exposure_value + correction > 8.0:
            self.set_controls(exposure_value=8.0)
        else:
            self.set_controls(exposure_value=self.exposure_value + correction)
        self.set_controls(exposure_value=self.exposure_value+correction)

    def calculate_brightness(self, image: np.array = None) -> float:
        """
        Calculates the brightness of the image.

        Args:
            image (np.array, optional): The image to calculate the brightness. If not provided, the image from camera will be used.

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
    
    def check_attributes(self) -> None:
        """
        Check if the attributes are valid.

        Check if the attributes are valid. Raises a ValueError if the attributes are invalid.

        Returns:
            None
        """
        if not (isinstance(self.resolution, tuple) and len(self.resolution) == 2 and all(isinstance(res, int) for res in self.resolution)):
            raise ValueError("The resolution attribute must be a tuple of two integers.")
        if not isinstance(self.framerate,int):
            raise ValueError("The framerate attribute must be an integer.")
        if not isinstance(self.turn_auto_exposure_on, bool):
            raise ValueError("The auto_exposure_on attribute must be a boolean.")
        if not isinstance(self.use_usb, bool):
            raise ValueError("The use_usb attribute must be a boolean.")
        else:
            if self.use_usb:
                self.check_attributes_usb()
            else:
                self.check_attributes_picamera()
    
    def check_attributes_usb(self) -> None:
        """
        Checks the attributes for USB camera

        Returns: 
            None
        """
        if not isinstance(self.saturation, float|int) or self.saturation < 0 or self.saturation > 200:
            raise ValueError("The USB saturation attribute must be a float in range [0; 200].")
        if not isinstance(self.sharpness, float|int) or self.sharpness < 0 or self.sharpness > 50:
            raise ValueError("The USB sharpness attribute must be a float in range [0; 50].")
        if not isinstance(self.brightness, float|int) or self.brightness < 30 or self.brightness > 255:
            raise ValueError("The USB brightness attribute must be a float in range [30; 255].")
        if not isinstance(self.contrast, float|int) or self.contrast < 0 or self.contrast > 10:
            raise ValueError("The USB contrast attribute must be a float in range [0; 10].")
        

    def check_attributes_picamera(self) -> None:
        """
        Checks the attributes for picamera

        Returns:
            None
        """
        if not isinstance(self.exposure_value, float|int) or self.exposure_value < -8.0 or self.exposure_value > 8.0:
            raise ValueError("The picamera2 exposure value attribute must be a float in range [-8.0; 8.0]")
        if not isinstance(self.saturation, float|int) or self.saturation < 0.0 or self.saturation > 32.0:
            raise ValueError("The picamera2 saturation attribute must be a float in range [0.0; 32.0]")
        if not isinstance(self.sharpness, float|int) or self.sharpness < 0.0 or self.sharpness > 16.0:
            raise ValueError("The picamera2 sharpness attribute must be a float in range [0.0; 16.0]")
        if not isinstance(self.brightness, float|int) or self.brightness< -1.0 or self.brightness > 1.0:
            raise ValueError("The picamera2 brightness attribute must be a float in range [-1.0; 1.0]")
        if not isinstance(self.contrast, float|int) or self.contrast < 0.0 or self.contrast > 32.0:
            raise ValueError("The picamera2 contrast attribute must be a float in range [0.0; 32.0]")
    
    def release(self):
        self.camera.release()
