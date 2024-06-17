from raspicam import Raspicam

def main():
    cam = Raspicam((1720, 720))
    cam.print_settings()
    cam.capture_img_and_save("test.jpg")

if __name__ == "__main__":
    main()