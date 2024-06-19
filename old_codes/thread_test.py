"""
A short test to see how threading works in Python.

Author: Josef Kahoun 
Date: 19.06.2024
"""

import threading
from raspicam import Raspicam
from trigger import Trigger
import cv2
import time

def start_stream(camera : Raspicam) -> None:
    print("jupii")
    """
    while True:
        image = camera.capture_img()
        cv2.imshow("Stream", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    """

def take_pictures(camera : Raspicam, folder_name : str) -> None:
    print("TAKE PICTURES")
    my_trigg = Trigger(camera=camera, folder_name=folder_name,num_of_pictures=5,
                       trigger_delay=3,times_between_pictures=3)
    my_trigg.trigg()

NUM_OF_THREADS = 2
functions = [start_stream, take_pictures]
def main():
    camera = Raspicam()
    arguments = [(camera,), (camera,"test_folder")]
    # create threads 
    threads = []
    for i in range(NUM_OF_THREADS):
        thread = threading.Thread(target=functions[i], args=arguments[i])
        threads.append(thread)
    thread = threading.Thread(target=functions[1],args=arguments[1])
    threads.append(thread)
    print("starting threads")
    for thread in threads:
        thread.start()
        time.sleep(1)
    print("joining threads")
    for thread in threads:
        thread.join()
    print("threads finished")
    camera.stop()
if __name__ == "__main__":
    main()