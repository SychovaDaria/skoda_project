"""
A module for imlementing various blur filters on an image.

Implemented filters:
    Average
    Gaussian
    Median
    Bilateral

Author: Josef Kahoun
Date: 4.7.2024
"""
import cv2
import numpy as np
from typing import Tuple

DEFAULT_KERNEL_SIZE = (2,2)
DEFAULT_MEDIAN_KERNEL_SIZE = 5
DEFAULT_BILATERAL_DIAMETER = 9
DEFAULT_BILATERAL_SIGMA_COLOR = 75 

class Filter:
    def __init__(self, kernel_size = DEFAULT_KERNEL_SIZE,median_kernel_size = DEFAULT_MEDIAN_KERNEL_SIZE,
                 bilat_diameter = DEFAULT_BILATERAL_DIAMETER, 
                 bilat_sigma_color = DEFAULT_BILATERAL_SIGMA_COLOR) -> None:
        # average and gaussian filter parameters
        self.kernel_size = kernel_size
        # median filter parameters
        self.median_kernel_size = median_kernel_size
        # bilateral filter parameters
        self.bilat_diameter = bilat_diameter
        self.bilat_sigma_color = bilat_sigma_color
        self.check_attributes()

    def average(self, img: np.array) -> np.array:
        """
        Apply the average filter on the image.

        Args:
            img (np.array): The image to apply the filter on.

        Returns:
            np.array: The image after applying the filter.
        """
        return cv2.blur(img, self.kernel_size)
    
    def gaussian(self, img: np.array) -> np.array:
        """
        Apply the gaussian filter on the image.

        Args:
            img (np.array): The image to apply the filter on.

        Returns:
            np.array: The image after applying the filter.
        """
        return cv2.GaussianBlur(img, self.kernel_size, 0)
    
    def median(self, img: np.array) -> np.array:
        """
        Apply the median filter on the image.

        Args:
            img (np.array): The image to apply the filter on.

        Returns:
            np.array: The image after applying the filter.
        """
        return cv2.medianBlur(img, self.median_kernel_size)

    def bilateral(self, img: np.array) -> np.array:
        """
        Apply the bilateral filter on the image.

        Args:
            img (np.array): The image to apply the filter on.

        Returns:
            np.array: The image after applying the filter.
        """
        return cv2.bilateralFilter(img, self.bilat_diameter, self.bilat_sigma_color, 75)

    def check_attributes(self) -> None:
        """
        Check if the attributes are valid.
        """
        if not isinstance(self.kernel_size, tuple) or len(self.kernel_size) != 2:
            raise ValueError("The kernel size must be a tuple of 2 integers.")
        if not all(isinstance(x, int) for x in self.kernel_size):
            raise ValueError("The kernel size must be a tuple of 2 integers.")
        if not isinstance(self.median_kernel_size, int):
            raise ValueError("The median kernel size must be an integer.")
        if not isinstance(self.bilat_diameter, int) or self.bilat_diameter <= 0:
            raise ValueError("The bilateral diameter must be a positive integer.")
        if not isinstance(self.bilat_sigma_color, int): #TODO: add range for sigma color
            raise ValueError("The bilateral sigma color must be an integer.")