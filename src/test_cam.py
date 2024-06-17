from raspicam import Raspicam

def main():
    cam = Raspicam((1720, 720))
    cam.print_settings()
    cam.start_cam()
    #cam.capture_img_and_save("test.jpg")
    #cam.stop_cam()
if __name__ == "__main__":
    main()