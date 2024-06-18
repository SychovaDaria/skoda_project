import cv2
from typing import List,Tuple

class Color_detector:
    def __init__(self, color: List[int, int, int],color_threshold : int,reg_of_interest: List[int, int, int, int],
                 min_width: int = 0, min_heigth : int = 0, min_area : int = 0):
        self.color = color
        self.color_threshold = color_threshold
        self.region_of_interest = reg_of_interest
        self.min_width = min_width
        self.min_heigth = min_heigth
        self.min_area = min_area

    def extract_color(self):
        pass

    def reduce_noise(self):
        pass