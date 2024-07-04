"""
Script to test edge detection.

Author: Josef Kahoun 
Date: 28.06.2024
"""
from edges import EdgeDetector
import cv2
import numpy as np
from skda_blur import Filter

def main():
    filt = Filter(kernel_size=(3,3))
    img = cv2.imread("../test_img/blob_test.jpg")
    img = filt.gaussian(img)
    edge_detector = EdgeDetector(min_val=10, max_val=50, min_value_of_votes=90, 
                                 min_length_of_straight_line=60,min_length=60,max_gap_between_lines=4,
                                 angle=90,angle_tolerance=180)
    fin_lines = edge_detector.get_lines(img)
    print(fin_lines)
    for line in fin_lines:
        x1,y1,x2,y2 = line
        cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.imshow("Original", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()