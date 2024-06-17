
import argparse
import time
import sys
import cv2
import numpy as np
from picamera2 import Picamera2
from tflite_runtime.interpreter import Interpreter

def run(model: str, label_map: str, min_confidence: float, width: int, height: int) -> None:
    """Continuously run inference on images acquired from the camera to detect objects.

    Args:
        model: Path to the TensorFlow Lite model file.
        label_map: Path to the label map file.
        min_confidence: Minimum confidence score for object detection.
        width: The width of the frame captured from the camera.
        height: The height of the frame captured from the camera.
    """

    # Load label map
    with open(label_map, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Load the TensorFlow Lite model
    interpreter = Interpreter(model_path=model)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']

    # Initialize the camera
    picam2 = Picamera2()
    camera_config = picam2.create_preview_configuration(main={"size": (width, height), "format": "RGB888"})
    picam2.configure(camera_config)
    picam2.start()

    # Continuously capture images from the camera and run inference
    while True:
        frame = picam2.capture_array()
        
        if frame is None:
            sys.exit('ERROR: Unable to read from camera. Please verify your camera settings.')

        # Resize and normalize the image
        rgb_image = cv2.resize(frame, (input_shape[1], input_shape[2]))
        input_data = np.expand_dims(rgb_image, axis=0)
        input_data = (np.float32(input_data) - 127.5) / 127.5

        # Perform inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Get detection results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects

        # Visualization parameters
        text_color = (0, 255, 0)  # Green for text
        box_color = (255, 0, 0)   # Red for bounding box
        font_size = 0.5
        font_thickness = 1

        # Loop over all detected objects
        for i in range(len(scores)):
            if scores[i] > min_confidence:
                # Get bounding box coordinates and scale them
                ymin, xmin, ymax, xmax = boxes[i]
                (left, right, top, bottom) = (xmin * width, xmax * width, ymin * height, ymax * height)
                left, right, top, bottom = int(left), int(right), int(top), int(bottom)

                # Get the label for the object
                label = labels[int(classes[i])]

                if label == 'phone':  # Check if the detected object is a phone
                    # Save the frame with the detected phone
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    cv2.imwrite(f'phone_detected_{timestamp}.jpg', frame)
                    
                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
                label_text = f'{label}: {int(scores[i] * 100)}%'
                cv2.putText(frame, label_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, font_size, text_color, font_thickness, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('Object Detection', frame)

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break

    picam2.stop()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model',
        help='Path to the TensorFlow Lite model file.',
        required=True)
    parser.add_argument(
        '--label_map',
        help='Path to the label map file.',
        required=True)
    parser.add_argument(
        '--min_confidence',
        help='The minimum confidence score for object detection.',
        type=float,
        default=0.5)
    parser.add_argument(
        '--frame_width',
        help='Width of frame to capture from camera.',
        type=int,
        default=640)
    parser.add_argument(
        '--frame_height',
        help='Height of frame to capture from camera.',
        type=int,
        default=480)
    args = parser.parse_args()

    run(args.model, args.label_map, args.min_confidence, args.frame_width, args.frame_height)

if __name__ == '__main__':
    main()
