import cv2
import numpy as np
from raspicam import Raspicam
from datetime import datetime

class EdgeDetectionCamera:
    def __init__(self, use_usb=False, save_path1=None, save_path2=None):
        # Initialize the Raspberry Pi camera module
        self.camera = Raspicam(use_usb=use_usb)
        self.save_path1 = save_path1
        self.save_path2 = save_path2

        # Initialize background subtractor
        self.back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

        # Initialize a counter for alternating between save paths
        self.photo_counter = 0

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
                            # Determine the save path based on the counter
                            save_path = self.save_path1 if self.photo_counter % 2 == 0 else self.save_path2
                                
                            # Save the image
                            filename = datetime.now().strftime("%d.%m.%Y_%H.%M.%S") + ".png"
                            self.camera.capture_img_and_save(filename=filename, folder_path=save_path)
                            print(f"Photo saved to {save_path}: {filename}")
                                
                            # Alternate the save path for the next photo
                            self.photo_counter += 1
                                
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
    save_path1 = "object2"  
    save_path2 = "object3"  
    camera = EdgeDetectionCamera(use_usb=False, save_path1=save_path1, save_path2=save_path2)
    camera.start_video_stream()

