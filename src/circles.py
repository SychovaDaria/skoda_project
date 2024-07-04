"""
A module for detecting circles in the scene.

Author: Josef Kahoun
Date: 4.7.2024
"""

import cv2
import numpy as np
from typing import List, Tuple

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

    Attributes:
        inverse_ratio (float): The inverse ratio of the accumulator resolution to the image resolution.
        min_dist_between_circles (int): The minimum distancea between the centers of the detected circles.
        accumulator_value (int): The accumulator threshold value for the circle centers at the detection stage.
        canny_param (int): The threshold value for the Canny edge detection.
        min_radius (int): The minimum radius of the circles to be detected.
        max_radius (int): The maximum radius of the circles to be detected.
    """

    def __init__(self, inverse_ratio: float = DEFAULT_INVERSE_RATIO,
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

    def detect_circles(self, img: np.array) -> List[Tuple[int, int, int]]:
        """
        Detect circles in the image.

        Args:
            img (np.array): The image to detect circles in.

        Returns:
            List[Tuple[int, int, int]]: A list of tuples containing the x and y coordinates of the circle centers and the radius.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, self.inverse_ratio, self.min_dist_between_circles,
                                   param1=self.canny_param, param2=self.accumulator_value,
                                   minRadius=self.min_radius, maxRadius=self.max_radius)
        if circles is None:
            circles = [[]]
        circles = np.uint8(np.around(circles))
        return circles[0]

    def update_attributes(self, inverse_ratio: float = None,
                          min_dist_between_circles: int = None,
                          accumulator_value: int = None, canny_param: int = None,
                          min_radius: int = None, max_radius: int = None) -> None:
        """
        Update the attributes of the CircleDetector.

        Args:
            inverse_ratio (float, optional): The inverse ratio of the accumulator resolution to the image resolution.
            min_dist_between_circles (int, optional): The minimum distancea between the centers of the detected circles.
            accumulator_value (int, optional): The accumulator threshold value for the circle centers at the detection stage.
            canny_param (int, optional): The threshold value for the Canny edge detection.
            min_radius (int, optional): The minimum radius of the circles to be detected.
            max_radius (int, optional): The maximum radius of the circles to be detected.
        """
        if inverse_ratio is not None:
            self.inverse_ratio = inverse_ratio
        if min_dist_between_circles is not None:
            self.min_dist_between_circles = min_dist_between_circles
        if accumulator_value is not None:
            self.accumulator_value = accumulator_value
        if canny_param is not None:
            self.canny_param = canny_param
        if min_radius is not None:
            self.min_radius = min_radius
        if max_radius is not None:
            self.max_radius = max_radius
        self.check_attributes()

    def check_attributes(self) -> None:
        """
        Check if the attributes are valid.
        """
        if self.inverse_ratio <= 0:
            raise ValueError("Inverse ratio must be greater than 0.")
        if self.min_dist_between_circles <= 0:
            raise ValueError("Minimum distance between circles must be greater than 0.")
        if self.accumulator_value <= 0:
            raise ValueError("Accumulator value must be greater than 0.")
        if self.canny_param <= 0:
            raise ValueError("Canny parameter must be greater than 0.")
        if self.min_radius <= 0:
            raise ValueError("Minimum radius must be greater than 0.")
        if self.max_radius <= 0 or self.max_radius < self.min_radius:
            raise ValueError("Maximum radius must be greater than 0 and greater or equal to minimum radius.")