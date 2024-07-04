"""
Module for edge detection in a picture.

Author: Josef Kahoun 
Date: 28.06.2024
"""

import cv2
import numpy as np
from typing import List, Tuple

#TODO: add docstrings
#TODO: test the class
#TODO: add type hints
#TODO: add variables to the class

DEFAULT_MIN_VALUE_OF_VOTES = 50
DEFAULT_MIN_LENGTH_OF_STRAIGHT_LINE = 50
DEFAULT_MAX_GAP_BETWEEN_LINES = 5
DEFAULT_MIN_LENGTH = 150
DEFAULT_ANGLE = 0
DEFAULT_ANGLE_TOLERANCE = 5

class EdgeDetector:
    """
    A class for edge detection in a picture.

    Args:
        min_val (int): The minimum value for edge detection.
        max_val (int): The maximum value for edge detection.
        min_value_of_votes (int): The minimum value of votes for line extraction.
        min_length_of_straight_line (int): The minimum length of a line for extraction.
        max_gap_between_lines (int): The maximum gap between lines for extraction.

    Attributes:
        min_val (int): The minimum value for edge detection.
        max_val (int): The maximum value for edge detection.
        min_value_of_votes (int): The minimum value of votes for line extraction.
        min_length_of_straight_line (int): The minimum length of a straight line for extraction.
        max_gap_between_lines (int): The maximum gap between lines for extraction.
    """
    def __init__(self, min_val: int, max_val: int, min_value_of_votes: int = DEFAULT_MIN_VALUE_OF_VOTES,
                 min_length_of_straight_line: int = DEFAULT_MIN_LENGTH_OF_STRAIGHT_LINE,
                 max_gap_between_lines: int = DEFAULT_MAX_GAP_BETWEEN_LINES,
                 min_length: int = DEFAULT_MIN_LENGTH, angle : int = DEFAULT_ANGLE,
                 angle_tolerance : int = DEFAULT_ANGLE_TOLERANCE) -> None:
        # edge detection parameters
        self.min_val = min_val
        self.max_val = max_val
        # straight line extraction parameters
        self.min_value_of_votes = min_value_of_votes
        self.min_length_of_straight_line = min_length_of_straight_line
        self.max_gap_between_lines = max_gap_between_lines
        # parameter for first contour extraction
        self.min_length = min_length
        # parameters for wanted lines
        self.angle = angle
        self.angle_tolerance = angle_tolerance
        self.check_attributes()

    def detect_edges(self, img: np.array) -> np.array:
        """
        Detects edges in an image using the Canny edge detection algorithm.

        Args:
                img (np.array): The image to detect edges in.

        Returns:
                np.array: The image with detected edges.
        """
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # convert to grayscale
        edges = cv2.Canny(gray_img, self.min_val, self.max_val)
        return edges
    
    def extract_straight_lines(self, img: np.array) -> np.array:
        """
        Extract straight lines from the image with detected edges using the Hough Line Transform.

        Args:
            img (np.array): The image with detected edges.

        Returns:
            np.array: The image with extracted straight lines.
        """
        lines = cv2.HoughLinesP(img, # image with lines
                                1, # distance resolution in pixels
                                np.pi/180, # radius resolution
                                self.min_value_of_votes, # min value of votes
                                minLineLength=self.min_length_of_straight_line, # min length of line in pixels
                                maxLineGap=self.max_gap_between_lines) # max gap between lines in pixels
        ret_lines = []
        if lines is not None:
            for line in lines:
                ret_lines.append(line[0]) # x0, y0, x1, y1
        return ret_lines

    def extract_connected_objects(self, img: np.array) -> Tuple[List,List]:
        """
        Extract connected objects from the image with detected edges.

        Args:
            img (np.array): The original image.

        Returns:
            list: List of bounding boxes of the connected objects.
            list: List of centroids of the connected objects.
        """
        bounding_boxes = []
        edges = self.detect_edges(img)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges)
        ret_centroids = []
        ret_img = np.zeros_like(edges)
        for i in range(1, num_labels):
            x, y, w, h, length = stats[i]
            if length > self.min_length:
                bounding_boxes.append(([x, y], [x+w,y+h])) # x0, y0, x1, y1
                ret_centroids.append((centroids[i][0], centroids[i][1]))
                ret_img = np.logical_or(ret_img, labels == i)

        return bounding_boxes,ret_centroids,np.uint8(ret_img*255)
    
    def get_lines(self, img : np.array) -> List[Tuple[int,int,int,int]]:
        """
        Get the lines from the img having the desired characteristics.

        Args:
            img (np.array): The image with detected edges.
        
        Returns:
            list: List of lines in the image.
        """
        _,_,edges = self.extract_connected_objects(img)
        lines = self.extract_straight_lines(edges)
        ret_lines = []
        for line in lines:
            x1,y1,x2,y2 = line
            cur_angle = np.arctan2(y2-y1,x2-x1)*180/np.pi
            if abs(cur_angle-self.angle) < self.angle_tolerance or abs(cur_angle-self.angle-180) < self.angle_tolerance or abs(cur_angle-self.angle+180) < self.angle_tolerance:
                ret_lines.append(line)
        return ret_lines


    def update_attributes(self, min_val: int = None, max_val: int = None, min_value_of_votes: int = None,
                         min_length_of_straight_line: int = None, max_gap_between_lines: int = None, 
                         min_length: int = None, angle: int = None, angle_tolerance: int = None):
        """
        Update the attributes of the EdgeDetector class.

        Args:
            min_val (int): The minimum value for edge detection.
            max_val (int): The maximum value for edge detection.
            min_value_of_votes (int): The minimum value of votes for line extraction.
            min_length_of_straight_line (int): The minimum length of a straight line for extraction.
            max_gap_between_lines (int): The maximum gap between lines for extraction.
            min_length (int): The minimum length of a line for first extraction.
            angle (int): The desired angle for line extraction.
            angle_tolerance (int): The tolerance for the desired angle.

        """
        if min_val is not None:
            self.min_val = min_val
        if max_val is not None:
            self.max_val = max_val
        if min_value_of_votes is not None:
            self.min_value_of_votes = min_value_of_votes
        if min_length_of_straight_line is not None:
            self.min_length_of_straight_line = min_length_of_straight_line
        if max_gap_between_lines is not None:
            self.max_gap_between_lines = max_gap_between_lines
        if min_length is not None:
            self.min_length = min_length
        if angle is not None:
            self.angle = angle
        if angle_tolerance is not None:
            self.angle_tolerance = angle_tolerance
        self.check_attributes()

    def check_attributes(self):
        """
        Check the current attributes of the EdgeDetector class and validate their correctness.
        """
        attributes = {
            'min_val': self.min_val,
            'max_val': self.max_val,
            'min_value_of_votes': self.min_value_of_votes,
            'min_length_of_straight_line': self.min_length_of_straight_line,
            'max_gap_between_lines': self.max_gap_between_lines,
            'min_length': self.min_length,
            'angle_tolerance': self.angle_tolerance
        }
        # Add attribute validation logic here
        for attr_name, attr_value in attributes.items():
            if not isinstance(attr_value, int) or attr_value <= 0:
                raise ValueError(f"{attr_name} attribute must be a positive integer.")
        if self.angle < 0 or self.angle > 180:
            raise ValueError("The angle must be between 0 and 180 degrees.")