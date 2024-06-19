"""
Test with GPIO of the rapsberry pi and threading

Author: Josef Kahoun
Date: 19.06.2024
"""
from gpiozero import DigitalInputDevice
import threading
from raspicam import Raspicam
from trigger import Trigger
import cv2
import time
from pynput import keyboard

BUTTON_PIN = 17
button = DigitalInputDevice(BUTTON_PIN)
running = True

def on_press(key):
    global running
    try:
        if key.char == 'q':
            print("Exiting the loop.")
            running = False
            # Stop listener
            return False
    except AttributeError:
        pass

# Start the keyboard listener
listener = keyboard.Listener(on_press=on_press)
listener.start()

def start_stream(camera : Raspicam) -> None:
    while True:
        image = camera.capture_img()
        cv2.imshow("Stream", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def start_gpio(camera : Raspicam, folder_name : str,button:DigitalInputDevice) -> None:
    global running
    print("START GPIO")
    threads = []
    while running:
        if button.value:
            print("Button pressed")
            thread = threading.Thread(target=take_pictures, args=(camera,folder_name))
            threads.append(thread)
            thread.start()
            time.sleep(1)
    for thread in threads:
        thread.join()
    print("end gpio")

def take_pictures(camera : Raspicam, folder_name : str) -> None:
    print("TAKE PICTURES")
    my_trigg = Trigger(camera=camera, folder_name=folder_name)
    my_trigg.trigg()

functions = [start_stream, start_gpio]


def main():
    camera = Raspicam()
    arguments = [(camera,), (camera,"test_folder",button)]
    threads = []
    for i in range(len(functions)):
        thread = threading.Thread(target=functions[i], args=arguments[i])
        threads.append(thread)
    # start the threads
    for thread in threads:
        thread.start()
    print("joining threads")
    for thread in threads:
        thread.join()
    print("threads finished")
    camera.stop()
if __name__ == "__main__":
    main()