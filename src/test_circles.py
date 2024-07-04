from circles import CircleDetector
from skda_blur import Filter
import cv2
import numpy as np

def main():
    circle_detector = CircleDetector(min_radius=20,min_dist_between_circles=50,max_radius=60,accumulator_value=100)
    filt = Filter(kernel_size=(7,7))
    img = cv2.imread("../test_img/eye.jpeg")
    img = filt.average(img)
    circs = circle_detector.detect_circles(img)
    for i in circs:
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
    cv2.imshow("img",img)
    cv2.waitKey(0)
if __name__ == "__main__":
    main()