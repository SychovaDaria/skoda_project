"""
Module for detecting and decoding barcodes and QR codes

Author: Josef Kahoun
Date: 11.7.2024
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict

class CodeDetector:
    """
    A class for detecting and decoding barcodes and QR codes.

    Args:
        find_qr_codes (bool): Whether to detect QR codes.
        find_barcodes (bool): Whether to detect barcodes.

    Attributes:
        find_qr_codes (bool): Whether to detect QR codes.
        find_barcodes (bool): Whether to detect barcodes.
        qr_detector (cv2.QRCodeDetector): The QR code detector object.
        barcode_detector (cv2.BarcodeDetector): The barcode detector object.
    """
    def __init__(self,find_qr_codes : bool = False, find_barcodes : bool = False) -> None:
        self.find_qr_codes = find_qr_codes
        self.find_barcodes = find_barcodes
        self.qr_detector = cv2.QRCodeDetector()
        self.barcode_detector = cv2.barcode.BarcodeDetector()

    def detect_codes(self, image: np.array) -> Dict[Dict[str, List]]:
        """
        Detect and decode codes in the image.

        Args:
            image (np.array): The image to detect codes in.

        Returns:
            tuple: A touple of list of found codes and list of their positions.
        """
        result = {'qr' : {}, 'barcode' : {}}
        if self.find_qr_codes:
            result['qr'] = self.detect_qr_codes(image)
        if self.find_barcodes:
            result['barcode'] = self.detect_barcodes(image)
        return result
    
    def detect_qr_codes(self, image: np.array) -> Dict[str, List]:
        """
        Detect and decode QR codes in the image.

        Args:
            image (np.array): The image to detect QR codes in.

        Returns:
            dict: A dictionary of decoded QR code, its position and straight QR code (binary matrix).
        """
        ret_qr, decoded_info, points, straigth_qr_code  = self.qr_detector.detectAndDecodeMulti(image)
        if not ret_qr:
            decoded_info = ()
            points = []
            straigth_qr_code = []
        return {'decoded_info' :decoded_info,'points': points, 'straight_qr_code':straigth_qr_code}
    
    def detect_barcodes(self, image: np.array) -> Dict[str, List]:
        """
        Detect and decode barcodes in the image.

        Args:
            image (np.array): The image to detect barcodes in.

        Returns:
            dict: A dictionary of decoded barcode, its position and type.
        """
        ret_barcode, decoded_info, decoded_type, points = self.barcode_detector.detectAndDecodeMulti(image)
        if not ret_barcode:
            decoded_info = ()
            points = []
            decoded_type = ()
        return {'decoded_info': decoded_info, 'points':points, 'decoded_type':decoded_type}