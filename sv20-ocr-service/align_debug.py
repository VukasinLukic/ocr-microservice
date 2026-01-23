import cv2
import json
import numpy as np
import os

# Path to the uploaded image
IMAGE_PATH = r"C:/Users/INSOMNIA/.gemini/antigravity/brain/d1794262-f73f-4c36-82d4-734b9a640f29/uploaded_image_1769104840108.png"
TEMPLATE_PATH = r"c:\Users\INSOMNIA\ocr-microservice\sv20-ocr-service\templates\sv20_template.json"
OUTPUT_PATH = r"C:/Users/INSOMNIA/.gemini/antigravity/brain/d1794262-f73f-4c36-82d4-734b9a640f29/debug_visual.jpg"

def draw_boxes():
    if not os.path.exists(IMAGE_PATH):
        print(f"Image not found: {IMAGE_PATH}")
        return

    # Load image
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("Failed to load image")
        return

    # Load template
    with open(TEMPLATE_PATH, 'r', encoding='utf-8') as f:
        template = json.load(f)

    print(f"Image shape: {img.shape}")
    
    # Simulate the preprocessing step that happens in server
    # We must replicate exactly what process_image does
    from processors.image_processor import ImageProcessor
    processor = ImageProcessor()
    
    # Preprocess (Deskew, etc.) - This might change the image dimensions/orientation!
    # Note: server.py calls: img = image_processor.preprocess_full_document(img)
    # We should do the same to see what the OCR engine sees.
    
    processed_img = processor.preprocess_full_document(img)
    print(f"Processed image shape: {processed_img.shape}")
    
    # Force convert to BGR for drawing if it's grayscale
    if len(processed_img.shape) == 2:
        vis_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
    else:
        vis_img = processed_img.copy()

    # Draw boxes
    for field in template['fields']:
        # Only page 1 for now since we process page 1
        if field.get('page', 1) != 1:
            continue
            
        coords = field['coordinates']
        x, y, w, h = coords['x'], coords['y'], coords['width'], coords['height']
        
        # Draw rectangle
        # Red color
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # Put text
        cv2.putText(vis_img, str(field['id']), (x, y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Save result
    cv2.imwrite(OUTPUT_PATH, vis_img)
    print(f"Saved debug image to {OUTPUT_PATH}")

if __name__ == "__main__":
    draw_boxes()
