import cv2
import numpy as np
import os
import sys
from PIL import Image, ImageOps

# Check and install piexif if needed
try:
    import piexif
except ImportError:
    print("📦 Installing required package: piexif...")
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "piexif"])
        import piexif
        print("✅ piexif installed successfully\n")
    except Exception as e:
        print(f"❌ Failed to install piexif: {e}")
        print("⚠️ Script will continue without metadata preservation...\n")
        piexif = None

# ==========================
# Configuration
# ==========================
TARGET_SIZE = 1024
MARGIN_X_RATIO = 0.1
MARGIN_Y_RATIO = 0.2
MARGIN_TOP_WITH_CHICKEN = 0.4
MARGIN_TOP_DEFAULT = 0.25
FALLBACK_CROP_RATIO = 0.7

# ==========================
# Utility Functions
# ==========================
def calculate_solidity(contour):
    """Calculate solidity of a contour (area/convex_hull_area)"""
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    return area / hull_area if hull_area > 0 else 0

# ==========================
# Board Detection
# ==========================
def detect_board_smart(image):
    """
    Detect cutting board in image using color-based and edge-based methods
    Returns: (x, y, w, h) bounding box or None
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Method 1: Color-based detection (white board)
    lower_white = np.array([0, 0, 160])
    upper_white = np.array([180, 60, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Morphological operations to clean up mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    mask = cv2.dilate(mask, np.ones((10, 10), np.uint8), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 5000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / h
        if 0.5 < aspect < 2.5 and calculate_solidity(c) > 0.7:
            candidates.append((area, (x, y, w, h)))

    # Method 2: Edge-based detection (fallback)
    if not candidates:
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), 2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < 8000:
                continue
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / h
            if 0.5 < aspect < 2.5:
                candidates.append((area, (x, y, w, h)))

    # Return largest candidate if it's significant enough
    if candidates:
        area, rect = max(candidates, key=lambda x: x[0])
        img_area = image.shape[0] * image.shape[1]
        if area < img_area * 0.1:
            return None
        return rect

    return None

def detect_chicken_direction(image, board_rect):
    """
    Detect which direction has more content (chicken) relative to board
    Returns: "top", "bottom", "left", "right", or "center"
    """
    if board_rect is None:
        return "center"

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    x, y, w, h = board_rect

    # Define regions around the board
    regions = {
        "top": edges[max(0, y - h//2):y, x:x+w],
        "bottom": edges[y+h:min(edges.shape[0], y+h+h//2), x:x+w],
        "left": edges[y:y+h, max(0, x-w//2):x],
        "right": edges[y:y+h, x+w:min(edges.shape[1], x+w+w//2)],
    }

    # Calculate edge density for each region
    scores = {k: np.sum(v > 0) / (v.size + 1) for k, v in regions.items()}
    return max(scores, key=scores.get)

# ==========================
# Image Processing
# ==========================
def process_single_image(input_path, output_path):
    """
    Process a single image: detect board, crop intelligently, resize, and preserve metadata
    """
    if os.path.exists(output_path):
        print(f"⏭️  Skip (exists): {os.path.basename(output_path)}")
        return True

    try:
        # 1. Load image and extract metadata
        with Image.open(input_path) as img:
            # Save original EXIF data
            exif_bytes = img.info.get("exif")
            
            # Auto-rotate based on EXIF orientation
            img = ImageOps.exif_transpose(img)
            
            # Rotate 90° if portrait (make it landscape)
            if img.height > img.width:
                img = img.rotate(90, expand=True, resample=Image.BICUBIC)
            
            # Convert to OpenCV format (BGR)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            open_cv_image = np.array(img)
            open_cv_image = open_cv_image[:, :, ::-1].copy()  # RGB to BGR

        # 2. Detect board and determine crop area
        h_img, w_img = open_cv_image.shape[:2]
        
        board_rect = detect_board_smart(open_cv_image)
        
        if board_rect:
            # Smart crop based on detected board
            x, y, w, h = board_rect
            direction = detect_chicken_direction(open_cv_image, board_rect)

            margin_x = int(w * MARGIN_X_RATIO)
            margin_y = int(h * MARGIN_Y_RATIO)
            margin_top = int(h * (MARGIN_TOP_WITH_CHICKEN if direction == "top" else MARGIN_TOP_DEFAULT))

            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_top)
            x2 = min(w_img, x + w + margin_x)
            y2 = min(h_img, y + h + margin_y)

            crop = open_cv_image[y1:y2, x1:x2]
        else:
            # Fallback: center crop
            size = int(min(h_img, w_img) * FALLBACK_CROP_RATIO)
            x1 = (w_img - size) // 2
            y1 = (h_img - size) // 2
            crop = open_cv_image[y1:y1+size, x1:x1+size]

        # 3. Resize to target size
        resized = cv2.resize(crop, (TARGET_SIZE, TARGET_SIZE), cv2.INTER_LANCZOS4)
        
        # 4. Convert back to PIL and save with metadata
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        result_img = Image.fromarray(rgb)
        
        save_kwargs = {
            "format": "JPEG",
            "quality": 95,
            "subsampling": 0
        }
        
        # Preserve and update EXIF metadata
        if exif_bytes and piexif:
            try:
                exif_dict = piexif.load(exif_bytes)
                # Set orientation to normal since we've handled rotation
                if "0th" in exif_dict:
                    exif_dict["0th"][piexif.ImageIFD.Orientation] = 1
                exif_bytes_updated = piexif.dump(exif_dict)
                save_kwargs["exif"] = exif_bytes_updated
            except Exception as e:
                print(f"    ⚠️  EXIF update failed: {e}, using original metadata")
                save_kwargs["exif"] = exif_bytes
        elif exif_bytes:
            # Use original EXIF if piexif not available
            save_kwargs["exif"] = exif_bytes
        
        result_img.save(output_path, **save_kwargs)
        print(f"✅ Processed: {os.path.basename(output_path)}")
        return True
        
    except Exception as e:
        print(f"❌ Error processing {os.path.basename(input_path)}: {e}")
        return False

# ==========================
# Batch Processing
# ==========================
def process_directory(input_dir, output_dir):
    """
    Process all images in input directory and save to output directory
    """
    # Validate input directory
    if not os.path.exists(input_dir):
        print(f"❌ Error: Input directory '{input_dir}' does not exist!")
        return
    
    if not os.path.isdir(input_dir):
        print(f"❌ Error: '{input_dir}' is not a directory!")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all supported image files
    supported_formats = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    files = [f for f in os.listdir(input_dir)
             if f.lower().endswith(supported_formats)]
    
    if not files:
        print(f"⚠️  No image files found in '{input_dir}'")
        return
    
    print(f"🔍 Found {len(files)} image(s) to process")
    print(f"📁 Input:  {input_dir}")
    print(f"📁 Output: {output_dir}")
    print(f"🎯 Target size: {TARGET_SIZE}x{TARGET_SIZE}px\n")
    
    # Process each file
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for i, filename in enumerate(files, 1):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        print(f"[{i}/{len(files)}] ", end="")
        
        if os.path.exists(output_path):
            skip_count += 1
        else:
            result = process_single_image(input_path, output_path)
            if result:
                success_count += 1
            else:
                error_count += 1
        
        # Progress update every 10 files
        if i % 10 == 0 and i < len(files):
            print(f"\n📊 Progress: {i}/{len(files)} files processed\n")
    
    # Final summary
    print(f"\n{'='*50}")
    print(f"🎉 Processing complete!")
    print(f"{'='*50}")
    print(f"✅ Processed: {success_count}")
    print(f"⏭️  Skipped:   {skip_count}")
    print(f"❌ Errors:    {error_count}")
    print(f"📊 Total:     {len(files)}")
    print(f"{'='*50}\n")

# ==========================
# Entry Point
# ==========================
if __name__ == "__main__":
    INPUT_DIR = "input"
    OUTPUT_DIR = "output"
    
    # Allow command-line arguments for directories
    if len(sys.argv) >= 2:
        INPUT_DIR = sys.argv[1]
    if len(sys.argv) >= 3:
        OUTPUT_DIR = sys.argv[2]
    
    print("=" * 50)
    print("🖼️  Auto-Crop Image Processor with Metadata Preservation")
    print("=" * 50)
    print()
    
    process_directory(INPUT_DIR, OUTPUT_DIR)