"""
Script to test edge detection.

Author: Josef Kahoun 
Date: 28.06.2024
"""
from edges import EdgeDetector
import cv2
import numpy as np

def main():
    img = cv2.imread("../test_img/phone.jpeg")
    edge_detector = EdgeDetector(min_val=200, max_val=400, min_value_of_votes=20, 
                                 min_length_of_line=30,max_gap_between_lines=5)
    edges = edge_detector.detect_edges(img)

    boxes,centroids,new_img = edge_detector.extract_connected_objects(img)
    straight_lines = edge_detector.extract_straight_lines(new_img)
    for line in straight_lines:
        x1,y1,x2,y2 = line
        cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 2)
    """
    for box in boxes:
        cv2.rectangle(img, box[0], box[1], (0, 255, 0), 2) 
    for centroid in centroids:
        cx, cy = centroid
        cv2.circle(img, (int(cx), int(cy)), 5, (0, 0, 255), -1)
    """    
    
    cv2.imshow("Edges", edges)
    cv2.imshow("Original", img)
    cv2.imshow("Connected objects", new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()