import os
import cv2
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Global variables for region selection
ref_point = []
cropping = False
image = None

# Function to handle mouse clicks for region selection
def click_and_crop(event, x, y, flags, param):
    global ref_point, cropping, image

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping:
            image_copy = image.copy()
            cv2.rectangle(image_copy, ref_point[0], (x, y), (0, 255, 0), 2)
            cv2.imshow("image", image_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cropping = False
        cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("image", image)

# Function to select a region on the image
def select_region(image_path):
    global image, ref_point
    ref_point = []
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return None

    clone = image.copy()
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow("image", image)
    cv2.setMouseCallback("image", click_and_crop)

    print(f"Press 'q' to confirm selection or 'r' to reset selection for {image_path}")

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            image = clone.copy()
            ref_point = []
            cv2.imshow("image", image)

        elif key == ord("q") and len(ref_point) == 2:
            print(f"Region selected: {ref_point[0]} to {ref_point[1]}")
            break

    cv2.destroyAllWindows()

    if len(ref_point) == 2:
        x1, y1 = ref_point[0]
        x2, y2 = ref_point[1]
        return (x1, y1, x2 - x1, y2 - y1)
    return None

# Loading images for training (without cropping)
def load_images_for_training(directory, target_size=(128, 128)):
    images = []
    image_paths = []

    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path)

            if img is not None:
                try:
                    img_resized = cv2.resize(img, target_size)
                    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                    images.append(img_gray.flatten())
                    image_paths.append(img_path)
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")
            else:
                print(f"Failed to load image: {img_path}")

    return np.array(images), image_paths

# Training One-Class SVM model on normal data
def train_one_class_svm(normal_data):
    if normal_data.size == 0:
        raise ValueError("Cannot train the model, as normal data is empty.")

    scaler = StandardScaler()
    normal_data_scaled = scaler.fit_transform(normal_data)

    model = OneClassSVM(gamma='auto', kernel='rbf', nu=0.1)
    model.fit(normal_data_scaled)

    print("Initial training completed.")
    return model, scaler

# Active learning with region selection
def active_learning_with_regions(anomaly_image_paths):
    selected_regions = []
    for img_path in anomaly_image_paths:
        print(f"Processing image: {img_path}")
        region = select_region(img_path)
        if region:
            print(f"Selected region for {img_path}: {region}")
            selected_regions.append((img_path, region))
    
    return selected_regions

# Retraining the model based on selected regions
def retrain_model_on_selected_regions(selected_regions, model, scaler):
    data_for_retraining = []
    for img_path, region in selected_regions:
        img = cv2.imread(img_path)
        if img is not None:
            x, y, w, h = region
            img_cropped = img[y:y+h, x:x+w]  # Crop based on selected region
            img_resized = cv2.resize(img_cropped, (128, 128))
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            data_for_retraining.append(img_gray.flatten())
        else:
            print(f"Error: Unable to load image {img_path}")

    data_for_retraining = np.array(data_for_retraining)
    data_scaled = scaler.transform(data_for_retraining)
    
    # Retraining with clustering
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(data_scaled)

    print("The model has been retrained based on the selected regions.")
    return kmeans

# Checking new images for anomalies using the trained model
def check_for_anomalies(directory, model, scaler, target_size=(128, 128)):
    image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is not None:
            img_resized = cv2.resize(img, target_size)
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            img_flattened = img_gray.flatten().reshape(1, -1)

            img_scaled = scaler.transform(img_flattened)
            prediction = model.predict(img_scaled)

            if prediction == -1:
                print(f"Anomaly detected in image: {img_path}")
            else:
                print(f"No anomaly in image: {img_path}")
        else:
            print(f"Failed to load image: {img_path}")

# Main process
if __name__ == "__main__":
    # Load normal images for initial training
    normal_data_dir = "object3_cut"
    normal_images, _ = load_images_for_training(normal_data_dir)
    model, scaler = train_one_class_svm(normal_images)

    # Path to anomaly images for active learning
    anomaly_data_dir = "object3_bad"
    anomaly_image_paths = [os.path.join(anomaly_data_dir, f) for f in os.listdir(anomaly_data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Active learning: selecting regions on anomaly images
    selected_regions = active_learning_with_regions(anomaly_image_paths)

    # Retraining the model based on the selected regions
    final_model = retrain_model_on_selected_regions(selected_regions, model, scaler)

    # Check new images for anomalies using the trained model
    new_images_dir = "object3_new"  # Directory with new images to check
    check_for_anomalies(new_images_dir, final_model, scaler)
