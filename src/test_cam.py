from raspicam import Raspicam
from color_blobs import ColorDetector
import cv2
import numpy as np
import copy

def main():
    cam = Raspicam()
    while True:
        img = cam.capture_img()
        cv2.imshow("image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()