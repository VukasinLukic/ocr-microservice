import json, base64, cv2, numpy as np

data = json.load(open('../debug_result_fixed.json', encoding='utf-8'))

for field in data['omr_rois']:
    if field['field_id'] == 24:  # bracni_status
        # Decode image
        img_data = field['roi_base64'].split(',')[1]
        nparr = np.frombuffer(base64.b64decode(img_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Get coordinates
        field_coords = field['coordinates']
        field_x = field_coords['x']
        field_y = field_coords['y']

        # Draw rectangles for each option
        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]  # Green, Red, Blue

        for i, opt in enumerate(field['omr_options']):
            # Absolute coordinates
            abs_x = opt['x']
            abs_y = opt['y']
            w = opt['width']
            h = opt['height']

            # Convert to relative
            rel_x = abs_x - field_x
            rel_y = abs_y - field_y

            # Apply center margin (30%)
            center_margin_x = int(w * 0.30)
            center_margin_y = int(h * 0.30)

            final_x = rel_x + center_margin_x
            final_y = rel_y + center_margin_y
            final_w = max(8, w - 2 * center_margin_x)
            final_h = max(8, h - 2 * center_margin_y)

            # Draw rectangle
            cv2.rectangle(img, (final_x, final_y), (final_x + final_w, final_y + final_h), colors[i], 2)

            # Add label
            cv2.putText(img, f"Opt {opt['label']}", (final_x, final_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)

        # Save annotated image
        cv2.imwrite('debug_bracni_status_annotated.png', img)
        print(f"Saved: debug_bracni_status_annotated.png")
        print(f"\nOption coordinates (AFTER 30% center margin):")

        for opt in field['omr_options']:
            abs_x = opt['x']
            abs_y = opt['y']
            w = opt['width']
            h = opt['height']
            rel_x = abs_x - field_x
            rel_y = abs_y - field_y

            center_margin_x = int(w * 0.30)
            center_margin_y = int(h * 0.30)
            final_x = rel_x + center_margin_x
            final_y = rel_y + center_margin_y
            final_w = max(8, w - 2 * center_margin_x)
            final_h = max(8, h - 2 * center_margin_y)

            print(f"  Opt {opt['label']}: ({final_x}, {final_y}) -> ({final_x + final_w}, {final_y + final_h}) [size: {final_w}x{final_h}]")
