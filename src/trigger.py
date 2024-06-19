"""
Module for picture aquisition from the camera.

This module is responsible for taking pictures from the camera and saving them to the disk. Multiple settings can be
adjusted, such as delay between the trigger and the start of the aquisition, the number of pictures to take, time
between pictures, folder where the pictures are saved, etc.

Author: Josef Kahoun
Date: 19.06.2024
Version: 0.1
"""

from raspicam import Raspicam
from typing import List
import os
import time

# default values
DEFAULT_TRIGGER_DELAY = 0
DEFAULT_NUM_OF_PICTURES = 1
DEFAULT_TIMES_BETWEEN_PICTURES = .5

class Trigger:
    """
    Object for handling aquisition of pictures from the camera.

    Args:
        camera (Raspicam): Object for handling the camera controls.
        folder_name (str): The folder where the pictures will be saved.
        trigger_delay (float, optional): The delay between the trigger and the start of the acquisition. Defaults to DEFAULT_TRIGGER_DELAY.
        num_of_pictures (int, optional): The number of pictures to take. Defaults to DEFAULT_NUM_OF_PICTURES.
        times_between_pictures (float|List[float], optional): The time between pictures. Can be a single float or a list of floats. If it is a float, the times between the pictures are 
        the same. Defaults to DEFAULT_TIMES_BETWEEN_PICTURES.
    
    Attributes:
        camera (Raspicam): Object for handling the camera controls.
        trigger_delay (float): The delay between the trigger and the start of the acquisition.
        num_of_pictures (int): The number of pictures to take.
        times_between_pictures (float|List[float]): The time between pictures. Can be a single float or a list of floats.
        folder_name (str): The folder where the pictures will be saved.
    """
    def __init__(self, camera : Raspicam,folder_name : str, trigger_delay : float = DEFAULT_TRIGGER_DELAY,
                 num_of_pictures : int = 1, 
                 times_between_pictures : float|List[float] = DEFAULT_TIMES_BETWEEN_PICTURES) -> None:
        self.camera = camera
        self.trigger_delay = trigger_delay
        self.num_of_pictures = num_of_pictures
        self.times_between_pictures = times_between_pictures
        self.folder_name = folder_name
        self.check_attributes()
    
    def check_attributes(self) -> None:
        """
        Check if the attributes are valid.

        Returns:
            None
        """
        if not isinstance(self.camera, Raspicam):
            raise ValueError("The camera attribute must be an instance of the Raspicam class.")
        if not isinstance(self.trigger_delay, float|int) or self.trigger_delay < 0:
            raise ValueError("The trigger_delay attribute must be a non-negative float.")
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
        if not (isinstance(self.folder_name, str) and os.path.isdir(self.folder_name)):
            raise ValueError("The folder_name attribute must be a string representing a valid directory path.")

    def set_config(self,  camera : Raspicam = None, folder_name : str = None, trigger_delay : float = None,
                   num_of_pictures: int = None, times_between_pictures : float|List[float] = None) -> None:
        """
        Set the configuration of the trigger.

        Args:
            camera (Raspicam, optional): Object for handling the camera controls.
            folder_name (str, optional): The folder where the pictures will be saved.
            trigger_delay (float, optional): The delay between the trigger and the start of the acquisition.
            num_of_pictures (int, optional): The number of pictures to take.
            times_between_pictures (float|List[float], optional): The time between pictures. Can be a single float or a list of floats.

        Returns:
            None
        """
        if camera is not None:
            self.camera = camera
        if folder_name is not None:
            self.folder_name = folder_name
        if trigger_delay is not None:
            self.trigger_delay = trigger_delay
        if num_of_pictures is not None:
            self.num_of_pictures = num_of_pictures
        if times_between_pictures is not None:
            self.times_between_pictures = times_between_pictures
        self.check_attributes()

    def trigg(self) -> None:
        """
        Take pictures from the camera using the settings in the attributes.

        Returns:
            None
        """
        current_time = time.strftime("%Y%m%d_%H%M%S")
        # wait for the trigger delay
        time.sleep(self.trigger_delay)
        # transform times_between_pictures to a list if it is a float
        if isinstance(self.times_between_pictures, float|int):
            times_between_pictures = [self.times_between_pictures] * (self.num_of_pictures - 1)
        else:
            times_between_pictures = self.times_between_pictures
        # capture the imgs
        for i in range(self.num_of_pictures):
            filename = f"{current_time}_{i}.jpg" # create the filename
            self.camera.capture_img_and_save(filename=filename, folder_path=self.folder_name)
            if i < self.num_of_pictures - 1:
                time.sleep(times_between_pictures[i])
        