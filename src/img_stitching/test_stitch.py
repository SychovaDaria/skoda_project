"""
A script for testing the Stitcher class
"""
import cv2
import numpy as np
from stitcher import Stitcher

def main():
    img1 = cv2.imread("../../test_img/stitching/img1.jpeg")
    img2 = cv2.imread("../../test_img/stitching/img2.jpeg")
    img3 = cv2.imread("../../test_img/stitching/img3.jpeg")
    stitcher = Stitcher()
    stat, img = stitcher.stitch_images([img1,img2,img3])#stitcher.stitch_images([img1, img2])
    if stat == 0:
        print("Stitched successfully")
        cv2.imwrite("../../test_img/stitching/stitched.jpeg", img)
    print("yupiii")
 
if __name__ == "__main__":
    main()