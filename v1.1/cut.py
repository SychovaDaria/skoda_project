import os
import cv2

class ImageCropper:
    def __init__(self, source_dir, target_dir, crop_region):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.crop_region = crop_region  # Crop region in the format (x, y, width, height)
        
        
        if not os.path.exists(self.target_dir):
            os.makedirs(self.target_dir)

    def crop_image(self, img, crop_region):
        x, y, w, h = crop_region
        return img[y:y+h, x:x+w]

    def process_images(self):
        for filename in os.listdir(self.source_dir):
            if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
                # Load the image
                img_path = os.path.join(self.source_dir, filename)
                img = cv2.imread(img_path)
                
                if img is not None:
                    # Crop the image
                    cropped_img = self.crop_image(img, self.crop_region)
                    
                    # Save the cropped image
                    target_path = os.path.join(self.target_dir, filename)
                    cv2.imwrite(target_path, cropped_img)
                    print(f"Saved cropped image: {target_path}")
                else:
                    print(f"Failed to load image: {img_path}")

# Usage example:
if __name__ == "__main__":
    source_dir = "object3"  
    target_dir = "object3_cut"  
    
    # Define the crop region: (x, y, width, height)
    crop_region = (1390, 180, 624, 512)  
    
    cropper = ImageCropper(source_dir, target_dir, crop_region)
    cropper.process_images()
