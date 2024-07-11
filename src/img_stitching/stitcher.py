"""
Module for stitching images together

Author: Josef Kahoun
Date: 11.7.2024
"""

import cv2
import numpy as np
from typing import List, Tuple

class Stitcher:
    """
    A class for stitching images together.

    Attributes:
        stitcher (cv2.Stitcher): The stitcher object.
    """
    def __init__(self) -> None:
        self.stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)

    def stitch_images(self, images: List[np.array]) -> Tuple[int, np.array]:
        """
        Stitch the images together.

        Args:
            images (list): The images to stitch together.

        Returns:
            tuple: The status of the stitching and the stitched image.
        """
        status, stitched = self.stitcher.stitch(images)
        return status, stitched

