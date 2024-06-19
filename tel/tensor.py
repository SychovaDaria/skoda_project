import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
import os
from raspicam import Raspicam

# trained model
model = load_model('phone_detector_model.h5')

#preprocessing
img_height, img_width = 150, 150

# Initialize 
camera = Raspicam()

def preprocess_image(image):
    image = cv2.resize(image, (img_height, img_width))
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def detect_phone(frame):
    processed_image = preprocess_image(frame)
    prediction = model.predict(processed_image)
    return prediction[0][0] > 0.5


os.makedirs('mobil', exist_ok=True)

last_detection_time = 0  # last detection time
check_interval = 20      # checking again

while True:
    current_time = time.time()
    frame = camera.capture_img()

    # Check if the current time is at least 20 seconds after the last detection
    if current_time - last_detection_time > check_interval:
        if detect_phone(frame):
            for i in range(3):  # Capture three images
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = os.path.join('mobil', f'phone_detected_{timestamp}.jpg')
                cv2.imwrite(filename, frame)
                print(f'Phone detected! Saved {filename}')
                time.sleep(0.5)  # Wait 0.5 
            last_detection_time = time.time()  # Update last detection time

    # stream
    cv2.imshow('Camera Stream', frame)

    # Break esc
    if cv2.waitKey(1) == 27:
        break

# Cleanup
camera.stop()  
cv2.destroyAllWindows()