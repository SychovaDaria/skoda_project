import cv2
import numpy as np
from typing import List,Tuple
import copy

class ColorDetector:
    """
    This class is used to detect color blobs in an image using segmentation.

    Segments a color in a regions of interest in an image and returns the bounding box of the detected blobs.
    Reduces noise in the image by comparing the area, width and heigth of the blobs. Can be used for camera trigger.

    Args:
            color (List[int, int, int]): The RGB color values to detect.
            color_threshold (int): The threshold value for color similarity.
            intensity_threshold (int): The threshold value for intensity similarity.
            reg_of_interest (List[int, int, int, int]): The region of interest (ROI) to search for color blobs.
                                                        If there is none specified, it searches the whole img.
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
                 reg_of_interest: List[Tuple[Tuple[int,int],Tuple[int,int]]] = None, min_width: int = 0, min_height: int = 0,
                 min_area: int = 0, box_line_color: Tuple[int, int, int] = (130, 130,0), box_line_width : float =2,
                 roi_line_color: Tuple[int, int, int] = (0,0,255), roi_line_width : float = 1) -> None: 
        self.color_reference = color_reference
        self.color_threshold = color_threshold
        self.intensity_threshold = intensity_threshold
        self.regions_of_interest = reg_of_interest
        self.min_width = min_width
        self.min_height = min_height
        self.min_area = min_area
        self.box_line_color = box_line_color
        self.box_line_width = box_line_width
        self.roi_line_color = roi_line_color
        self.roi_line_width = roi_line_width

    def extract_color(self,img : np.array) -> np.ndarray:
        """
        Extracts the color blobs from the image using the color reference.
        """
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
                
    def set_parameters(self, color_reference: Tuple[int, int, int], color_threshold: float = None, intensity_threshold: float = None,
                       reg_of_interest: Tuple[int, int, int, int] = None, min_width: int = None, min_height: int = None,
                       min_area: int = None, box_line_color: Tuple[int, int, int] = None, box_line_width : float = None,
                       roi_line_color: Tuple[int, int, int] = None, roi_line_width : float = None) -> None:
        """
        Sets the parameters for the color detector.
        """
        self.color_reference = color_reference
        if color_threshold is not None:
            self.color_threshold = color_threshold
        if intensity_threshold is not None:
            self.intensity_threshold = intensity_threshold
        if reg_of_interest is not None:
            self.regions_of_interest = reg_of_interest
        if min_width is not None:
            self.min_width = min_width
        if min_height is not None:
            self.min_height = min_height
        if min_area is not None:
            self.min_area = min_area
        if box_line_color is not None:
            self.box_line_color = box_line_color
        if box_line_width is not None:
            self.box_line_width = box_line_width
        if roi_line_color is not None:
            self.roi_line_color = roi_line_color
        if roi_line_width is not None:
            self.roi_line_width = roi_line_width
    
    def get_blobs(self, img : np.array) ->  List[Tuple[Tuple[int,int],Tuple[int,int]]]:
        """
        Returns the bounding boxes of the detected blobs in the regions of itnerest img.
        """
        if self.regions_of_interest is None:
            self.regions_of_interest = [((0,0),(img.shape[1],img.shape[0]))]
        final_boxes = []
        for roi in self.regions_of_interest:
            cropped_img = img[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]
            tmp_mask = self.reduce_noise(self.extract_color(cropped_img))
            tmp_boxes = self.get_bounding_boxes(tmp_mask)
            for box in tmp_boxes:
                box = ((box[0][0]+roi[0][0], box[0][1]+roi[0][1]), (box[1][0]+roi[0][0], box[1][1]+roi[0][1]))
                final_boxes.append(box)
        return final_boxes
    
    def draw_boxes(self, img:np.array)-> np.array:
        """
        Draws the bounding boxes of the detected blobs in the image.
        """
        ret_img = copy.deepcopy(img)
        boxes = self.get_blobs(img)
        for box in boxes:
            cv2.rectangle(ret_img, box[0], box[1], self.box_line_color, self.box_line_width)
        return ret_img
    
    def draw_rois(self, img : np.array) -> np.array:
        """
        Draws the regions of interest (ROIs) on the image.
        """
        ret_img = copy.deepcopy(img)
        for roi in self.regions_of_interest:
            cv2.rectangle(ret_img, roi[0], roi[1], self.roi_line_color, self.roi_line_width)
        return ret_img