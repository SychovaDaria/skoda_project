"""
A module for detecting circles in the scene.

Author: Josef Kahoun
Date: 4.7.2024
"""

import cv2
import numpy as np

DEFAULT_INVERSE_RATIO = 1.5
DEFAULT_MIN_DIST_BETWEEN_CIRCLES = 50
DEFAULT_CANNY_PARAM = 50
DEFAULT_ACCUMULATOR_VALUE = 100
DEFAULT_MIN_RADIUS = 10
DEFAULT_MAX_RADIUS = 60

#TODO: add type hints and docstrings

class CircleDetector:
    """
    A class for detecting circles in the scene.
    """
    def __init__(self, inverse_ratio : float = DEFAULT_INVERSE_RATIO,
                 min_dist_between_circles: int = DEFAULT_MIN_DIST_BETWEEN_CIRCLES,
                 accumulator_value: int = DEFAULT_ACCUMULATOR_VALUE, canny_param: int = DEFAULT_CANNY_PARAM,
                 min_radius: int = DEFAULT_MIN_RADIUS, max_radius: int = DEFAULT_MAX_RADIUS) -> None:
        self.inverse_ratio = inverse_ratio
        self.accumulator_value = accumulator_value
        self.canny_param = DEFAULT_CANNY_PARAM
        self.min_dist_between_circles = min_dist_between_circles
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.check_attributes()

    def detect_circles(self, img: np.array) -> np.array:
        """
        Detect circles in the image.

        Args:
            img (np.array): The image to detect circles in.

        Returns:
            np.array: The image with detected circles.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, self.inverse_ratio, self.min_dist_between_circles,
                                   param1=self.canny_param, param2=self.accumulator_value,
                                   minRadius=self.min_radius, maxRadius=self.max_radius)
        if circles is None:
            circles = [[]]
        circles = np.uint8(np.around(circles))
        return circles[0]
        
    def draw_circles(self) -> np.array:
        """
        Draw the detected circles on the image.

        Returns:
            np.array: The image with the detected circles.
        """
        

    def check_attributes(self) -> None:
        """
        Check if the attributes are valid.
        """
        pass