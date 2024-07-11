"""
A quick test of the QR and barcode detection classes.
"""

import cv2
import numpy as np
from codes_detector import CodeDetector

def main():
    # Load the image
    bar_img = cv2.imread('../test_img/barcode.jpeg')
    qr_img = cv2.imread('../test_img/qrcode.jpeg')
    # Create a detector object
    detector = CodeDetector(find_qr_codes=True, find_barcodes=True)
    # Detect codes
    result = detector.detect_codes(qr_img)
    # Print the result
    print(result)

if __name__ == '__main__':
    main()