import cv2
import numpy as np
from raspicam import Raspicam
from datetime import datetime

class EdgeDetectionCamera:
    def __init__(self, use_usb=False, save_path=None):
        # Initialize the Raspberry Pi camera module
        self.camera = Raspicam(use_usb=use_usb)
        self.save_path = save_path

        # Initialize background subtractor
        self.back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

    def detect_edges(self, img):
        # Apply background subtraction to get the foreground mask
        fg_mask = self.back_sub.apply(img)

        # Extract only the foreground from the original image
        fg_img = cv2.bitwise_and(img, img, mask=fg_mask)

        # Convert the image to grayscale
        gray = cv2.cvtColor(fg_img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to the image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Perform edge detection using Canny algorithm
        edges = cv2.Canny(blurred, 50, 150)

        return edges, fg_mask

    def start_video_stream(self):
        motion_detected = False
        still_frame_count = 0

        while True:
            # Capture an image from the camera
            img = self.camera.capture_img()

            if img is not None:
                # Perform edge detection and get the foreground mask
                edges, fg_mask = self.detect_edges(img)

                # Analyze the center region of the frame
                height, width = fg_mask.shape
                center_region = fg_mask[height//3:2*height//3, width//3:2*width//3]
                movement_in_center = np.sum(center_region) / 255  # Normalize movement detection

                if movement_in_center > 300:  # Threshold to detect significant movement
                    motion_detected = True
                    still_frame_count = 0
                else:
                    if motion_detected:
                        still_frame_count += 1
                        if still_frame_count > 10:  # Threshold for stabilization (about 10 frames)
                            # Save the image if the object has stopped in the center
                            filename = datetime.now().strftime("%d.%m.%Y_%H.%M.%S") + ".png"
                            self.camera.capture_img_and_save(filename=filename, folder_path=self.save_path)
                            print(f"Photo saved: {filename}")
                            motion_detected = False
                            still_frame_count = 0

                # Display the original image and the edge-detected image
                cv2.imshow("Video Stream", img)
                cv2.imshow("Edges", edges)

            # Break the loop and close windows if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources and close OpenCV windows
        cv2.destroyAllWindows()

# Usage example:
if __name__ == "__main__":
    save_path = "object2"  # Replace with your desired save path
    camera = EdgeDetectionCamera(use_usb=False, save_path=save_path)
    camera.start_video_stream()
