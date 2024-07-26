"""
A quick test of the trigger class

Author: Josef Kahoun
Date: 19.06.2024
"""
from trigger import Trigger
from raspicam import Raspicam
import cv2

def main():
    cam = Raspicam()
    my_trigg = Trigger(cam, "test_folder")
    while True:
        img = cam.capture_img()
        cv2.imshow("Stream",img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("capture img")
            my_trigg.trigg()
if __name__ == "__main__":
    main()