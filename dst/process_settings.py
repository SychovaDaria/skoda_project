"""
Module for classes containing settings for the AI training and detection, and the trigger module.

Author: Josef Kahoun
Date: 7. 8. 2024
"""

import os
from typing import Dict

DEFAULT_CONF_THRESHOLD = 0.7
DEFAULT_NUM_OF_IMAGES = 1
DEFAULT_DELAY = 0


class AiSettings():
    def __init__(self):
        self.end_process = False
        # the ai model settings
        self.start_trigger = False
        self.conf_thr = DEFAULT_CONF_THRESHOLD
        self.training_data_folder = None
        self.model_path = None
        #self.img_height = 150
        #self.img_width = 150

    def is_start_trigger(self) -> bool:
        """
        Checks if the trigger module should be started.

        Returns:
            bool: True if the trigger module should be started, False otherwise.
        """
        return self.start_trigger

    def set_end_message(self) -> None:
        """
        Sets the end message for the AI process.

        This message will be sent to the AI process to break the loop and join the main process.
        """
        self.end_process = True

    def is_end_message(self) -> bool:
        """
        Checks if the message is the end message.

        Returns:
            bool: True if this is the end message, False otherwise.
        """
        return self.end_process
        
    def update_settings(self, new_settings : Dict) -> None:
        """
        Update the AI settings with new values.

        Args:
            new_settings (dict): Dictionary containing the new settings.
        """
        if 'conf_thr' in new_settings:
            self.conf_thr = new_settings['conf_thr']
        if 'training_data_folder' in new_settings:
            self.training_data_folder = new_settings['training_data_folder']
        if 'model_path' in new_settings:
            self.model_path = new_settings['model_path']
        #if 'img_height' in new_settings:
        #    self.img_height = new_settings['img_height']
        #if 'img_width' in new_settings:
        #    self.img_width = new_settings['img_width']
