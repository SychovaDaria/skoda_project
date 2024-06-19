"""
A quick test of the trigger class

Author: Josef Kahoun
Date: 19.06.2024
"""
from trigger import Trigger
from raspicam import Raspicam

def main():
    cam = Raspicam()
    my_trigg = Trigger(cam, "test_folder", trigger_delay = 5, num_of_pictures = 5, times_between_pictures = 2)
    while True:
        if input() == 'q':
            print("Starting the trigger")
            my_trigg.trigg()
            print("Trigger finished")
            break
if __name__ == "__main__":
    main()