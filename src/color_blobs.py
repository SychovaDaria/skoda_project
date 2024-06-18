import cv2
import numpy as np
from typing import List,Tuple
import copy

class Color_detector:
    """
    This class is used to detect color blobs in an image using segmentation.

    Segments a color in a regions of interest in an image and returns the bounding box of the detected blobs.
    Reduces noise in the image by comparing the area, width and heigth of the blobs. Can be used for camera trigger.

    Args:
            color (List[int, int, int]): The RGB color values to detect.
            color_threshold (int): The threshold value for color similarity.
            intensity_threshold (int): The threshold value for intensity similarity.
            reg_of_interest (List[int, int, int, int]): The region of interest (ROI) to search for color blobs.
            min_width (int, optional): The minimum width of a color blob. Defaults to 0.
            min_height (int, optional): The minimum height of a color blob. Defaults to 0.
            min_area (int, optional): The minimum area of a color blob. Defaults to 0.
    
    Attributes:
            color_reference (List[int, int, int]): The RGB color values to detect.
            color_threshold (float): The threshold value for color similarity.
            intensity_threshold (float): The threshold value for normed intensity value, ranges 0.0 to 1.0
            reg_of_interest (List[int, int, int, int]): The region of interest (ROI) to search for color blobs.
                                                        List of starting points and end points
            min_width (int, optional): The minimum width of a color blob. Defaults to 0.
            min_height (int, optional): The minimum height of a color blob. Defaults to 0.
            min_area (int, optional): The minimum area of a color blob. Defaults to 0.

    """
    def __init__(self, color_reference: Tuple[int, int, int], color_threshold: float, intensity_threshold: float,
                 reg_of_interest: List[Tuple[Tuple[int,int],Tuple[int,int]]], min_width: int = 0, min_height: int = 0,
                 min_area: int = 0):    
        self.color_reference = color_reference
        self.color_threshold = color_threshold
        self.intensity_threshold = intensity_threshold
        self.region_of_interest = reg_of_interest
        self.min_width = min_width
        self.min_height = min_height
        self.min_area = min_area

    def extract_color(self,img : np.array) -> np.ndarray:
        """
        Extracts the color blobs from the image using the color reference.
        """
        # TODO: implement the regiens of interest
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        normed_img = img/np.linalg.norm(img, axis = 2)[:,:,np.newaxis]
        mask = np.linalg.norm(img, axis = 2) > self.intensity_threshold # intensity threshold
        normed_ref = self.color_reference/np.linalg.norm(self.color_reference) # normalize the reference
        mask = np.logical_and(mask, np.linalg.norm(normed_img-normed_ref,axis = 2) < self.color_threshold)
        return mask      

    def reduce_noise(self, mask : np.ndarray):
        """
        Reduces noise in the mask by comparing the area, width and heigth of the blobs.
        """
        components = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        stats = components[2]
        final_mask = copy.deepcopy(mask)
        final_mask[:] = 0
        good_labels = []
        for label in range(1,len(stats)):#0 is background
            if (stats[label, cv2.CC_STAT_WIDTH] > self.min_width and stats[label, cv2.CC_STAT_HEIGHT] > self.min_height
                and stats[label, cv2.CC_STAT_AREA] > self.min_area):
                good_labels.append(label)
        for label in good_labels:
            final_mask = np.logical_or(final_mask, components[1] == label)
        return final_mask
    
    def get_bounding_boxes(self, mask : np.ndarray) -> List[Tuple[Tuple[int,int],Tuple[int,int]]]:
        """
        Returns the bounding box of the detected blobs in the mask.
        """
        stats = cv2.connectedComponentsWithStats(mask.astype(np.uint8))[2]
        bounding_boxes = []
        for label in range(1,len(stats)):
            start_point = (stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP])
            end_point = (start_point[0] + stats[label, cv2.CC_STAT_WIDTH], start_point[1] + stats[label, cv2.CC_STAT_HEIGHT])
            bounding_boxes.append((start_point, end_point))
        return bounding_boxes
            
    def get_blobs(self, img : np.array) ->  List[Tuple[Tuple[int,int],Tuple[int,int]]]:
        """
        Returns the bounding box of the detected blobs in the img.
        """
        mask = self.reduce_noise(self.extract_color(img))
        bounding_boxes = self.get_bounding_boxes(mask)
        return bounding_boxes
    
    def set_parameters(self, color_reference: Tuple[int, int, int], color_threshold: float = None, intensity_threshold: float = None,
                       reg_of_interest: Tuple[int, int, int, int] = None, min_width: int = None, min_height: int = None,
                       min_area: int = None):
        """
        Sets the parameters for the color detector.
        """
        self.color_reference = color_reference
        if color_threshold is not None:
            self.color_threshold = color_threshold
        if intensity_threshold is not None:
            self.intensity_threshold = intensity_threshold
        if reg_of_interest is not None:
            self.region_of_interest = reg_of_interest
        if min_width is not None:
            self.min_width = min_width
        if min_height is not None:
            self.min_height = min_height
        if min_area is not None:
            self.min_area = min_area
    