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
                 num_of_pictures : int = 1, 
                 times_between_pictures : float|List[float] = DEFAULT_TIMES_BETWEEN_PICTURES,
                 trigger_mode: TriggerModes = TriggerModes.ALWAYS) -> None:
        self.trigger_delay = trigger_delay
        self.num_of_pictures = num_of_pictures
        self.times_between_pictures = times_between_pictures
        self.folder_name = folder_name
        self.trigger_mode = trigger_mode
        self.last_trigger_signal = False
        self.last_trigger_time = time.time()
        self.capturing = False
        self.delay_number = 0
        self.check_attributes()
    
    def check_attributes(self) -> None:
        """
        Checks if the attributes are valid.

        Returns:
            None
        """
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
        if self.folder_name is not None and (not (isinstance(self.folder_name, str) and os.path.isdir(self.folder_name))):
            raise ValueError("The folder_name attribute must be a string representing a valid directory path.")
        if not isinstance(self.trigger_mode, TriggerModes):
            raise ValueError("The trigger_mode attribute must be an instance of the TriggerModes enum.")

    def set_config(self, folder_name : str = None, trigger_delay : float = None,
                   num_of_pictures: int = None, times_between_pictures : float|List[float] = None,
                   trigger_mode: TriggerModes = None) -> None:
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
        if num_of_pictures is not None:
            self.num_of_pictures = num_of_pictures
        if times_between_pictures is not None:
            self.times_between_pictures = times_between_pictures
            # transform times_between_pictures to a list if it is a float
            if isinstance(self.times_between_pictures, float|int):
                self.times_between_pictures = [self.times_between_pictures] * (self.num_of_pictures - 1)
        if trigger_mode is not None:
            self.trigger_mode = trigger_mode
        
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
            (self.trigger_mode == TriggerModes.ALWAYS and trigger_signal) or 
            (self.trigger_mode == TriggerModes.RISING_EDGE and trigger_signal and not self.last_trigger_signal) or 
            (self.trigger_mode == TriggerModes.FALLING_EDGE and not trigger_signal and self.last_trigger_signal)
            )
        ):
            self.capturing = True
            self.last_trigger_time = time.time()
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
            if not self.wait_for_delay(self.last_trigger_time, self.trigger_delay):
                return
            self.delay_number += 1
            self.last_trigger_time = time.time()
            self.save_img(img,self.delay_number-1)
        # the delays between the pictures
        if self.delay_number <= self.num_of_pictures:
            wait_time = self.times_between_pictures[self.delay_number - 1]
            if not self.wait_for_delay(self.last_trigger_time, wait_time):
                return
            self.last_trigger_time = time.time()
            self.save_img(img,self.delay_number-1)
            self.delay_number += 1
        else:
            self.capturing = False
        
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
        current_time = time.strftime("%m%d_%H%M%S")
        filename = f"{current_time}_{num_of_img}.jpg"
        cv2.imwrite(self.folder_name+"/"+filename,img)
        
    '''
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
    '''