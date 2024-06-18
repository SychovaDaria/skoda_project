#from raspicam import Raspicam
from color_blobs import ColorDetector
import cv2
import numpy as np
import copy

def main():
    img = cv2.imread("../test_img/blob_test.jpg")
    color_ref = [252, 57, 3]
    color_threshold = 0.25
    intensity_threshold = 0.05
    print(img.shape)
    reg_of_interest = [((0,0),(100,100)), ((100,100),(200,200))]
    min_width = 10
    min_height = 10
    min_area = 100
    detector = ColorDetector(color_ref, color_threshold, intensity_threshold, None, 
                              min_width, min_height, min_area)
    boxed_img = detector.draw_boxes(img)
    
    cv2.imshow("normal img", img)
    cv2.imshow("boxed img", detector.draw_rois(boxed_img))
    cv2.waitKey(0)

if __name__ == "__main__":
    main()