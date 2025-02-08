from PIL import Image, ImageFilter, ImageDraw
import numpy as np

def image_to_matrix(image_path: str, threshold: int = 128, debug: bool = False) -> np.ndarray:
    # Load image and convert to grayscale
    img = Image.open(image_path).convert('L')
    arr = np.array(img)
    if debug:
        print("[DEBUG] Loaded image with shape:", arr.shape)

    # Binarize: True for dark (pixel value < threshold), False for light
    binary = arr < threshold
    if debug:
        print("[DEBUG] Binarized image, unique values:", np.unique(binary))

    # Crop white padding: find rows and columns that contain any dark pixel
    rows = np.any(binary, axis=1)  # True if any dark pixel in the row
    cols = np.any(binary, axis=0)  # True if any dark pixel in the column

    if not np.any(rows) or not np.any(cols):
        raise ValueError('No QR code found in image')

    # Find the exact boundaries of the QR code
    top = int(np.argmax(rows))
    bottom = int(len(rows) - 1 - np.argmax(rows[::-1]))
    left = int(np.argmax(cols))
    right = int(len(cols) - 1 - np.argmax(cols[::-1]))
    if debug:
        print(f"[DEBUG] Cropping coordinates: top={top}, bottom={bottom}, left={left}, right={right}")

    # Calculate the actual dimensions of the QR code
    qr_height = bottom - top + 1
    qr_width = right - left + 1

    # Ensure the cropped region is square by expanding the smaller dimension
    if qr_height > qr_width:
        diff = qr_height - qr_width
        left = max(0, left - diff // 2)
        right = min(arr.shape[1] - 1, right + (diff + 1) // 2)
    elif qr_width > qr_height:
        diff = qr_width - qr_height
        top = max(0, top - diff // 2)
        bottom = min(arr.shape[0] - 1, bottom + (diff + 1) // 2)

    cropped = binary[top:bottom + 1, left:right + 1]
    if debug:
        print("[DEBUG] Cropped image shape:", cropped.shape)
    cropped_gray = arr[top:bottom + 1, left:right + 1]

    # Find the top-left finder pattern
    def find_finder_pattern_size():
        # Search in the first 25% of the image for the finder pattern
        search_height = cropped.shape[0] // 4
        search_width = cropped.shape[1] // 4
        
        # Find first dark pixel from top-left
        for i in range(search_height):
            for j in range(search_width):
                if cropped[i, j]:
                    start_row, start_col = i, j
                    
                    # Measure horizontal width of the finder pattern
                    width = 0
                    j_temp = start_col
                    while j_temp < cropped.shape[1] and cropped[start_row, j_temp]:
                        width += 1
                        j_temp += 1
                    
                    # Measure vertical height of the finder pattern
                    height = 0
                    i_temp = start_row
                    while i_temp < cropped.shape[0] and cropped[i_temp, start_col]:
                        height += 1
                        i_temp += 1
                    
                    # The finder pattern should be roughly square
                    if abs(width - height) <= max(width, height) * 0.2:  # Allow 20% difference
                        return (width + height) / 2  # Average of width and height
                    
        raise ValueError("Could not find a valid finder pattern")

    # Get the finder pattern size and calculate module size (finder pattern is 7 modules)
    finder_size = find_finder_pattern_size()
    one_module_size = finder_size / 7.0
    
    # Keep the module size as a float for more precise grid alignment
    if debug:
        print(f"[DEBUG] Detected finder pattern size: {finder_size:.2f} pixels")
        print(f"[DEBUG] Calculated module size: {one_module_size:.2f} pixels")

    # Calculate the module count based on the precise module size
    module_count_h = int(round(cropped.shape[0] / one_module_size))
    module_count_w = int(round(cropped.shape[1] / one_module_size))
    module_count = min(module_count_h, module_count_w)
    
    if debug:
        print(f"[DEBUG] Estimated module count: {module_count}")
        if module_count < 21 or module_count > 177:  # Valid QR code sizes
            print("[WARNING] Unusual module count detected. QR codes should be between 21x21 and 177x177")

    # Build the QR matrix using the precise module size
    qr_matrix = np.zeros((module_count, module_count), dtype=float)
    for i in range(module_count):
        for j in range(module_count):
            r_start = int(i * one_module_size)
            r_end = int((i + 1) * one_module_size) if i < module_count - 1 else cropped_gray.shape[0]
            c_start = int(j * one_module_size)
            c_end = int((j + 1) * one_module_size) if j < module_count - 1 else cropped_gray.shape[1]

            region = cropped_gray[r_start:r_end, c_start:c_end]
            if region.size == 0:
                continue
            # Compute the average grayscale value of the cell
            avg = np.mean(region)
            qr_matrix[i, j] = avg

    if debug:
        print("[DEBUG] QR Matrix built.")
        # Create a visualization overlaying the grid on the cropped, binarized image
        vis = np.where(cropped, 0, 255).astype(np.uint8)
        base_img = Image.fromarray(vis).convert('L').convert('RGBA')
        # Create an overlay for grid lines with transparency
        overlay = Image.new('RGBA', base_img.size, (255, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Draw grid lines using the precise module size
        for i in range(module_count + 1):
            y = i * one_module_size
            draw.line([(0, y), (cropped.shape[1], y)], fill=(255, 0, 0, 255), width=1)
        for j in range(module_count + 1):
            x = j * one_module_size
            draw.line([(x, 0), (x, cropped.shape[0])], fill=(255, 0, 0, 255), width=1)
            
        combined = Image.alpha_composite(base_img, overlay)
        combined.save('grid_visualization.png')
        print("[DEBUG] Grid visualization saved to grid_visualization.png")

    # Convert average matrix to binarized matrix using threshold
    binarized_matrix = (qr_matrix < threshold).astype(int)
    return binarized_matrix


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python qr_image.py <path_to_image>")
    else:
        image_path = sys.argv[1]
        print(f"Testing image: {image_path}")
        matrix = image_to_matrix(image_path, debug=True)
        print("QR Code Matrix:")
        for row in matrix:
            print("".join(["██" if cell else "  " for cell in row]))
        print("Finished processing image.") 