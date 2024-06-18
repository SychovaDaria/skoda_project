from raspicam import Raspicam
from color_blobs import ColorDetector
import cv2
import numpy as np
import copy
import time

def main():
    cam = Raspicam()
    detector = ColorDetector(color_reference = (80,40,30), color_threshold = 0.3, intensity_threshold = 0.1)
    detector.set_parameters(min_width = 50, min_height = 50, min_area = 1000, reg_of_interest=[((100,100),(1400,1400))])
    while True:
        img = cam.capture_img()
        mask = detector.reduce_noise(detector.extract_color(img))
        new_img = copy.deepcopy(img)
        new_img[~mask] = 0
        cv2.imshow("image", detector.draw_rois(detector.draw_boxes(new_img)))
        time.sleep(0.1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()