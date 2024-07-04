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
    filter = Filter()
    img = cv2.imread("../test_img/blob_test.jpg")
    img = filter.bilateral(img)
    edge_detector = EdgeDetector(min_val=200, max_val=400, min_value_of_votes=60, 
                                 min_length_of_straight_line=50,max_gap_between_lines=4,angle=45)
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