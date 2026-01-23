import cv2
import numpy as np
import json
import os

# Path to the user uploaded image (assuming I need to move it or it's accessible)
# I will use the path from the metadata if I can access it, or just use a placeholder
# for the user to run, but since I am the agent, I can access the file system.
# The user uploaded image path is: 
# C:/Users/INSOMNIA/.gemini/antigravity/brain/d1794262-f73f-4c36-82d4-734b9a640f29/uploaded_image_1769099691823.png

IMAGE_PATH = r"C:/Users/INSOMNIA/.gemini/antigravity/brain/d1794262-f73f-4c36-82d4-734b9a640f29/uploaded_image_1769099691823.png"

def check_image():
    if not os.path.exists(IMAGE_PATH):
        print(f"Image not found at {IMAGE_PATH}")
        return

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("Failed to load image.")
        return

    print(f"Image dimensions: {img.shape}")
    
    # Load template to compare expected coordinates
    template_path = r"c:\Users\INSOMNIA\ocr-microservice\sv20-ocr-service\templates\sv20_template.json"
    with open(template_path, 'r', encoding='utf-8') as f:
        template = json.load(f)
    
    print(f"Template Name: {template['document']['name']}")
    print(f"Expected DPI: {template['document']['expected_dpi']}")
    
    # Check a specific field, e.g., broj_indeksa (id 3)
    field = next(f for f in template['fields'] if f['id'] == 3)
    coords = field['coordinates']
    print(f"Field 'broj_indeksa' coords: {coords}")
    
    h, w = img.shape[:2]
    padding = 5
    x, y, width, height = coords['x'], coords['y'], coords['width'], coords['height']
    
    
    # Simulate processing
    from processors.image_processor import ImageProcessor
    processor = ImageProcessor()
    
    # Preprocess
    print("Running preprocess_full_document...")
    processed_img = processor.preprocess_full_document(img)
    print(f"Processed image shape: {processed_img.shape}")
    
    h, w = processed_img.shape[:2]
    padding = 5
    x, y, width, height = coords['x'], coords['y'], coords['width'], coords['height']
    
    y1 = max(0, y - padding)
    y2 = min(h, y + height + padding)
    
    print(f"Calculated ROI y range for processed image: {y1} to {y2}")
    
    if y1 >= y2:
        print("ERROR: ROI is empty because y1 >= y2")
    else:
        print("SUCCESS: ROI is valid.")


if __name__ == "__main__":
    check_image()
