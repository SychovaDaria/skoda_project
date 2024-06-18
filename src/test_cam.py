#from raspicam import Raspicam
from color_blobs import Color_detector
import cv2
import numpy as np
import copy

def main():
    img = cv2.imread("../test_img/blob_test.jpg")
    color_ref = [252, 57, 3]
    color_threshold = 0.25
    intensity_threshold = 0.05
    reg_of_interest = [0, 0, 100, 100]
    min_width = 10
    min_height = 10
    min_area = 100
    detector = Color_detector(color_ref, color_threshold, intensity_threshold, reg_of_interest, 
                              min_width, min_height, min_area)
    bounding_boxes = detector.get_blobs(img)
    for box in bounding_boxes:
        print(box)
        cv2.rectangle(img, (box[0]), (box[1]), (0, 255, 0), 2)
    mask_with_noise = detector.extract_color(img)
    img_with_noise = copy.deepcopy(img)
    img_with_noise[~mask_with_noise] = 0

    mask_without_noise = detector.reduce_noise(mask_with_noise)
    img_without_noise = copy.deepcopy(img)
    img_without_noise[~mask_without_noise] = 0
    cv2.imshow("with noise", img_with_noise)
    cv2.imshow("normal img", img)
    cv2.imshow("without noise", img_without_noise)
    #cv2.imshow("Blobs", img)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()