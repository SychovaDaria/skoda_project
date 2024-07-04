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
                                 min_length_of_straight_line=50,max_gap_between_lines=4)
    edges = edge_detector.detect_edges(img)

    boxes,centroids,new_img = edge_detector.extract_connected_objects(img)
    straight_lines = edge_detector.extract_straight_lines(new_img)
    print(straight_lines)
    for line in straight_lines:
        x1,y1,x2,y2 = line
        cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 2)
    
    cv2.imshow("Edges", edges)
    cv2.imshow("Original", img)
    cv2.imshow("Connected objects", new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()