import cv2
import numpy as np
import os
from PIL import Image
import piexif

def detect_white_board_smart(image):
    """
    Deteksi papan putih dengan prioritas menghindari area ayam
    Strategi: Cari area putih yang berbentuk persegi/persegi panjang
    """
    original = image.copy()
    height, width = image.shape[:2]
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Range untuk warna papan putih (lebih ketat)
    # Papan putih biasanya memiliki saturation rendah dan value tinggi
    lower_white = np.array([0, 0, 200])   # H: any, S: very low, V: very high
    upper_white = np.array([180, 40, 255])
    
    # Mask untuk warna putih
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Operasi morfologi untuk membersihkan noise dan menyatukan area papan
    kernel_open = np.ones((5, 5), np.uint8)
    kernel_close = np.ones((30, 30), np.uint8)  # Kernel besar untuk menyatukan area papan
    
    # Buka untuk hilangkan noise kecil (bulu ayam yang terdeteksi sebagai putih)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel_open)
    
    # Tutup untuk menyatukan area papan yang terputus
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel_close)
    
    # Temukan semua kontur
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Kandidat papan
    board_candidates = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter berdasarkan area (papan harus cukup besar)
        min_area = (height * width) * 0.1  # Minimal 10% dari gambar
        max_area = (height * width) * 0.8  # Maksimal 80% dari gambar
        
        if area < min_area or area > max_area:
            continue
        
        # Dapatkan bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter berdasarkan aspect ratio (papan cenderung persegi panjang)
        aspect_ratio = w / h
        if not (0.3 <= aspect_ratio <= 3.0):
            continue
        
        # Filter berdasarkan solidity (kekompakan bentuk)
        solidity = calculate_solidity(contour)
        if solidity < 0.6:
            continue
        
        # Filter berdasarkan rectangularity (seberapa mirip dengan rectangle)
        rect_area = w * h
        rectangularity = area / rect_area if rect_area > 0 else 0
        if rectangularity < 0.7:
            continue
        
        # Skor confidence berdasarkan multiple factors
        confidence = (solidity * 0.4 + rectangularity * 0.4 + 
                     min(1.0, area / (height * width * 0.3)) * 0.2)
        
        board_candidates.append({
            'contour': contour,
            'bbox': (x, y, w, h),
            'area': area,
            'confidence': confidence,
            'solidity': solidity,
            'rectangularity': rectangularity
        })
    
    if not board_candidates:
        return None
    
    # Pilih kandidat dengan confidence tertinggi
    best_candidate = max(board_candidates, key=lambda x: x['confidence'])
    x, y, w, h = best_candidate['bbox']
    
    print(f"📐 Papan terdeteksi: {w}x{h} (conf: {best_candidate['confidence']:.2f})")
    
    return (x, y, w, h)

def remove_chicken_from_mask(mask, image):
    """
    Hilangkan area ayam dari mask papan berdasarkan texture dan shape analysis
    """
    # Convert to grayscale untuk texture analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Deteksi edges - ayam biasanya memiliki banyak edges (bulu, dll)
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate edges untuk mendapatkan area yang lebih jelas
    edges_dilated = cv2.dilate(edges, np.ones((10, 10), np.uint8), iterations=2)
    
    # Area dengan banyak edges kemungkinan adalah ayam (bukan papan)
    chicken_like_areas = cv2.bitwise_and(mask, edges_dilated)
    
    # Kurangi area yang seperti ayam dari mask papan
    clean_mask = cv2.subtract(mask, chicken_like_areas)
    
    # Isi lubang yang mungkin terbentuk
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, np.ones((20, 20), np.uint8))
    
    return clean_mask

def calculate_solidity(contour):
    """Hitung solidity (area / convex hull area)"""
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    return area / hull_area if hull_area > 0 else 0

def detect_board_by_corners(image):
    """
    Fallback: Deteksi papan berdasarkan sudut-sudutnya
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Gunakan corner detection
    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 30)
    
    if corners is None:
        return None
    
    corners = np.int32(corners)
    
    # Cari rectangle dari corners
    if len(corners) >= 4:
        # Simple approach: gunakan min/max coordinates
        x_coords = corners[:, 0, 0]
        y_coords = corners[:, 0, 1]
        
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        w = x_max - x_min
        h = y_max - y_min
        
        # Filter reasonable size
        if w > 100 and h > 100 and w < image.shape[1] * 0.9 and h < image.shape[0] * 0.9:
            return (x_min, y_min, w, h)
    
    return None

def smart_crop_with_chicken_detection(image, board_rect):
    """
    Crop yang smart dengan mempertimbangkan posisi ayam
    """
    x, y, w, h = board_rect
    img_height, img_width = image.shape[:2]
    
    # Default margins
    margin_x = int(w * 0.1)
    margin_y_top = int(h * 0.3)
    margin_y_bottom = int(h * 0.1)
    
    # Deteksi apakah ada ayam di atas papan
    has_chicken_above = detect_chicken_above_board(image, board_rect)
    
    # Deteksi apakah ayam sangat besar dan overlap dengan papan
    chicken_overlap = detect_chicken_overlap(image, board_rect)
    
    if chicken_overlap:
        print("⚠️ Ayam overlap dengan papan, gunakan crop konservatif")
        # Gunakan crop yang lebih ketat
        margin_x = int(w * 0.05)
        margin_y_top = int(h * 0.1)
        margin_y_bottom = int(h * 0.05)
    elif has_chicken_above:
        print("🐔 Ayam terdeteksi di atas papan, sesuaikan margin atas")
        margin_y_top = int(h * 0.5)  # Beri lebih banyak space di atas
    else:
        print("✅ Tidak ada ayam mengganggu, gunakan margin normal")
    
    # Calculate final crop coordinates
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y_top)
    x2 = min(img_width, x + w + margin_x)
    y2 = min(img_height, y + h + margin_y_bottom)
    
    return (x1, y1, x2, y2)

def detect_chicken_above_board(image, board_rect):
    """
    Deteksi apakah ada ayam di area atas papan
    """
    x, y, w, h = board_rect
    
    # Area di atas papan untuk dianalisis
    above_height = int(h * 0.8)
    above_y1 = max(0, y - above_height)
    above_area = image[above_y1:y, x:x+w]
    
    if above_area.size == 0:
        return False
    
    # Analisis warna dan texture area atas
    hsv_above = cv2.cvtColor(above_area, cv2.COLOR_BGR2HSV)
    
    # Ayam biasanya memiliki warna yang bukan putih murni
    non_white_mask = cv2.inRange(hsv_above, np.array([0, 30, 0]), np.array([180, 255, 200]))
    
    # Analisis texture dengan variance
    gray_above = cv2.cvtColor(above_area, cv2.COLOR_BGR2GRAY)
    texture_variance = np.var(gray_above)
    
    # Jika ada cukup banyak non-white pixels atau texture tinggi, kemungkinan ada ayam
    non_white_ratio = np.sum(non_white_mask > 0) / non_white_mask.size
    
    return non_white_ratio > 0.2 or texture_variance > 500

def detect_chicken_overlap(image, board_rect):
    """
    Deteksi apakah ayam sangat besar dan overlap dengan area papan
    """
    x, y, w, h = board_rect
    
    # Area papan
    board_area = image[y:y+h, x:x+w]
    
    # Analisis variasi warna di dalam area papan
    hsv_board = cv2.cvtColor(board_area, cv2.COLOR_BGR2HSV)
    
    # Deteksi area yang bukan putih (kemungkinan ayam)
    non_white_in_board = cv2.inRange(hsv_board, np.array([0, 50, 0]), np.array([180, 255, 255]))
    non_white_ratio = np.sum(non_white_in_board > 0) / non_white_in_board.size
    
    # Jika banyak area non-putih di dalam papan, kemungkinan ayam overlap
    return non_white_ratio > 0.3

def preserve_metadata(original_path, new_image):
    """Pertahankan metadata dari gambar original"""
    try:
        original_pil = Image.open(original_path)
        exif_dict = {}
        if 'exif' in original_pil.info:
            exif_dict = piexif.load(original_pil.info['exif'])
        
        new_image_rgb = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
        new_pil = Image.fromarray(new_image_rgb)
        
        if exif_dict:
            exif_bytes = piexif.dump(exif_dict)
            return new_pil, exif_bytes
        else:
            return new_pil, None
            
    except Exception as e:
        print(f"⚠️ Warning: Tidak bisa mempertahankan metadata: {e}")
        new_image_rgb = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(new_image_rgb), None

def auto_crop(input_path, output_path):
    try:
        if os.path.exists(output_path):
            print(f"⏭️ Skip (sudah ada): {os.path.basename(output_path)}")
            return

        img = cv2.imread(input_path)
        if img is None:
            print(f"⚠️ Tidak bisa baca file: {input_path}")
            return

        print(f"\n📁 Processing: {os.path.basename(input_path)}")
        print(f"📐 Original size: {img.shape[1]}x{img.shape[0]}")

        # Step 1: Initial crop untuk hapus border putih eksternal
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
        coords = cv2.findNonZero(mask)

        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            # Beri padding kecil
            padding = 15
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img.shape[1] - x, w + 2 * padding)
            h = min(img.shape[0] - y, h + 2 * padding)
            
            img = img[y:y+h, x:x+w]
            print(f"📏 Initial crop: {w}x{h}")

        # Step 2: Deteksi papan dengan approach smart
        board_rect = detect_white_board_smart(img)
        
        if board_rect is None:
            print("🔄 Coba corner-based detection...")
            board_rect = detect_board_by_corners(img)

        if board_rect is not None:
            # Gunakan smart crop yang mempertimbangkan ayam
            x1, y1, x2, y2 = smart_crop_with_chicken_detection(img, board_rect)
            cropped = img[y1:y2, x1:x2]
            print(f"✅ Smart crop: [{x1},{y1}] to [{x2},{y2}]")
        else:
            # Fallback: center crop yang aman
            print("⚠️ Papan tidak terdeteksi, gunakan safe center crop")
            height, width = img.shape[:2]
            crop_size = int(min(width, height) * 0.7)  # 70% dari sisi terpendek
            x_center = (width - crop_size) // 2
            y_center = (height - crop_size) // 2
            cropped = img[y_center:y_center+crop_size, x_center:x_center+crop_size]
            print(f"🔄 Safe center crop: {crop_size}x{crop_size}")

        # Resize dan save
        if cropped.size > 0:
            resized = cv2.resize(cropped, (512, 512), interpolation=cv2.INTER_LANCZOS4)
            pil_image, exif_data = preserve_metadata(input_path, resized)
            
            if exif_data:
                pil_image.save(output_path, 'JPEG', quality=95, exif=exif_data)
            else:
                pil_image.save(output_path, 'JPEG', quality=95)

            print(f"💾 Saved: {os.path.basename(output_path)} (512x512)")
        else:
            print(f"❌ Error: Area crop kosong")

    except Exception as e:
        print(f"❌ Error processing {input_path}: {str(e)}")

def process_dir(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    total = len(files)
    print(f"🧩 Processing {total} files...")

    for i, file in enumerate(files, 1):
        in_path = os.path.join(input_dir, file)
        out_path = os.path.join(output_dir, file)
        auto_crop(in_path, out_path)

        if i % 10 == 0:
            print(f"📦 Processed {i}/{total}")

    print(f"✅ Done. Total: {total} files.")

if __name__ == "__main__":
    input_dir = "E:\WeightPrediction\ImageJPG"
    output_dir = "output"
    # input_dir = "test-input"
    # output_dir = "test-output"
    
    if os.path.isdir(input_dir):
        process_dir(input_dir, output_dir)
    else:
        print("❌ Folder 'input' tidak ditemukan.")