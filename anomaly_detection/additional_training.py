import os
import cv2

# Variables for cropping region coordinates
ref_point = []
cropping = False
image = None

# Function to handle mouse events for selecting a region
def click_and_crop(event, x, y, flags, param):
    global ref_point, cropping, image

    # Start selecting the region on left mouse button press
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True

    # Update the rectangle as the mouse moves
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping:
            image_copy = image.copy()
            cv2.rectangle(image_copy, ref_point[0], (x, y), (0, 255, 0), 2)
            cv2.imshow("image", image_copy)

    # Finalize region selection when the left mouse button is released
    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cropping = False

        # Draw a green rectangle around the selected region
        cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("image", image)

# Function to select a region on an image interactively
def select_region(image_path):
    global image, ref_point
    ref_point = []  # Reset coordinates
    image = cv2.imread(image_path)

    # Check if the image was successfully loaded
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None

    clone = image.copy()
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow("image", image)
    cv2.setMouseCallback("image", click_and_crop)

    print(f"Press 'q' to confirm selection or 'r' to reset the selection for {image_path}")

    # Wait for user to select a region or close the window
    while True:
        key = cv2.waitKey(1) & 0xFF

        # Reset the selection if 'r' is pressed
        if key == ord("r"):
            image = clone.copy()
            ref_point = []
            cv2.imshow("image", image)

        # Confirm the selection and move to the next image if 'q' is pressed
        elif key == ord("q") and len(ref_point) == 2:
            print(f"Region selected: {ref_point[0]} to {ref_point[1]}")
            break

    cv2.destroyAllWindows()

    # If a region was selected, return the coordinates
    if len(ref_point) == 2:
        x1, y1 = ref_point[0]
        x2, y2 = ref_point[1]
        return (x1, y1, x2 - x1, y2 - y1)  # Return coordinates and region size
    return None

# Function to process all images in a directory
def process_images_in_directory(directory):
    images = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    total_images = len(images)
    print(f"Found {total_images} images in {directory}")

    for i, filename in enumerate(images):
        img_path = os.path.join(directory, filename)
        print(f"Processing image {i+1}/{total_images}: {img_path}")

        # Select a region from the current image
        selected_region = select_region(img_path)

        if selected_region is not None:
            print(f"Selected region for {filename}: {selected_region}")
        else:
            print(f"No region selected for {filename}")
        
        # Debug message to show when moving to the next image
        if i < total_images - 1:
            print(f"Moving to the next image ({i+2}/{total_images})...\n")
        else:
            print("All images have been processed.")

# Main function to execute the process
if __name__ == "__main__":
    # Set the directory path containing the images
    directory_path = "object3_bad"

    # Process all images in the specified directory
    process_images_in_directory(directory_path)
