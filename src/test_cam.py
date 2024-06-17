from raspicam import Raspicam

def main():
    cam = Raspicam((1720, 720))
    cam.print_settings()
    #cam.capture_img_and_save("test.jpg")
    #cam.stop_cam()
    cam.start_preview()
    while True:
        image = cam.capture_img()
if __name__ == "__main__":
    main()