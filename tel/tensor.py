import cv2
import numpy as np
from tensorflow.keras.models import load_model
#from picamera2 import Picamera2
import time
import os
from raspicam import Picamera2

# trained model
model = load_model('phone_detector_model.h5')

# params
img_height, img_width = 150, 150

# cam initializace
# picam2 = Picamera2()
# camera_config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
# picam2.configure(camera_config)
# picam2.start()
camera = Paspicam()

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

while True:
    frame = camera.capture_img()
    if detect_phone(frame):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join('mobil', f'phone_detected_{timestamp}.jpg')
        cv2.imwrite(filename, frame)
        print(f'Phone detected! Saved {filename}')

    cv2.imshow('Camera Stream', frame)

    if cv2.waitKey(1) == 27:  #esc 
        break

camera.stop()
cv2.destroyAllWindows()
