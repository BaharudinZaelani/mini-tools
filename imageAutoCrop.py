import cv2
import numpy as np
import os
import piexif
import tempfile
from PIL import Image, ImageOps

# ==========================
# Utility & helper
# ==========================
def calculate_solidity(contour):
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    return area / hull_area if hull_area > 0 else 0

def preserve_metadata(original_path, new_image):
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
        return new_pil, None
    except Exception as e:
        print(f"⚠️ Tidak bisa mempertahankan metadata: {e}")
        new_image_rgb = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(new_image_rgb), None

# ==========================
# Smart detection functions
# ==========================
def detect_board_smart(image):
    """
    Deteksi area papan putih dengan robust method:
    kombinasi HSV mask + deteksi tepi + analisis area paling terang.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Mask untuk warna putih terang (saturasi rendah, value tinggi)
    lower_white = np.array([0, 0, 160])
    upper_white = np.array([180, 60, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Morphological filter
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    mask = cv2.dilate(mask, np.ones((10, 10), np.uint8), iterations=1)

    # Cari kontur papan
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 5000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / h
        if 0.5 < aspect < 2.5:
            solidity = calculate_solidity(c)
            if solidity > 0.7:
                candidates.append((area, (x, y, w, h)))
    candidates.sort(reverse=True)

    # Jika gagal deteksi, coba fallback dengan tepi
    if not candidates:
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < 8000:
                continue
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / h
            if 0.5 < aspect < 2.5:
                candidates.append((area, (x, y, w, h)))
        candidates.sort(reverse=True)

    if candidates:
        area, rect = candidates[0]
        # Jika area terlalu kecil dibanding gambar, anggap gagal
        img_area = image.shape[0] * image.shape[1]
        if area < img_area * 0.1:
            return None
        return rect
    return None

def detect_chicken_direction(image, board_rect):
    """
    Analisa distribusi tekstur (edge density) di sekitar papan
    untuk menentukan apakah ayam lebih ke atas atau ke samping.
    """
    if board_rect is None:
        return 'center'

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    x, y, w, h = board_rect

    regions = {
        'top': edges[max(0, y - h//2):y, x:x+w],
        'bottom': edges[y+h:min(edges.shape[0], y + h + h//2), x:x+w],
        'left': edges[y:y+h, max(0, x - w//2):x],
        'right': edges[y:y+h, x+w:min(edges.shape[1], x + w + w//2)],
    }

    scores = {k: np.sum(v > 0) / (v.size + 1) for k, v in regions.items()}
    direction = max(scores, key=scores.get)
    return direction

# ==========================
# Core crop logic
# ==========================
def auto_crop(input_path, output_path):
    # Skip jika sudah ada file output
    if os.path.exists(output_path):
        print(f"⏭️ Skip (sudah ada): {os.path.basename(output_path)}")
        return

    img = cv2.imread(input_path)
    if img is None:
        print(f"⚠️ Tidak bisa baca: {input_path}")
        return

    h_img, w_img = img.shape[:2]

    # Target output size
    TARGET_SIZE = 1024

    # Deteksi papan
    board_rect = detect_board_smart(img)
    if board_rect is not None:
        x, y, w, h = board_rect
        direction = detect_chicken_direction(img, board_rect)

        margin_x = int(w * 0.1)
        margin_y = int(h * 0.2)
        margin_top = int(h * (0.4 if direction == 'top' else 0.25))

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_top)
        x2 = min(w_img, x + w + margin_x)
        y2 = min(h_img, y + h + margin_y)

        crop = img[y1:y2, x1:x2]
        print(f"✅ Detected board ({direction}) at {x},{y} {w}x{h}")
    else:
        # Fallback ke tengah (square)
        print(f"⚠️ Gagal deteksi papan: {os.path.basename(input_path)}")

        size = int(min(h_img, w_img) * 0.7)
        x1 = (w_img - size) // 2
        y1 = (h_img - size) // 2

        crop = img[y1:y1 + size, x1:x1 + size]

    # Resize ke 1024x1024 (high quality)
    resized = cv2.resize(
        crop,
        (TARGET_SIZE, TARGET_SIZE),
        interpolation=cv2.INTER_LANCZOS4
    )

    # Simpan dengan metadata
    pil_image, exif_data = preserve_metadata(input_path, resized)
    if exif_data:
        pil_image.save(
            output_path,
            'JPEG',
            quality=95,
            subsampling=0,
            exif=exif_data
        )
    else:
        pil_image.save(
            output_path,
            'JPEG',
            quality=95,
            subsampling=0
        )

    print(f"💾 Saved: {os.path.basename(output_path)} ({TARGET_SIZE}x{TARGET_SIZE})")
    print("---")

def ensure_landscape(image_path, output_path=None):
    img = Image.open(image_path)
    width, height = img.size

    # Cek orientasi
    if height > width:
        print("Portrait detected → rotating...")
        img = img.rotate(90, expand=True)
    else:
        print("Already landscape")

    if output_path:
        img.save(output_path)

    return img


# ==========================
# Batch processing
# ==========================
def process_dir(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    files = [
        f for f in os.listdir(input_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    total = len(files)
    print(f"🧩 Processing {total} files...")

    for i, f in enumerate(files, 1):
        in_path = os.path.join(input_dir, f)
        out_path = os.path.join(output_dir, f)

        # ==========================
        # Check orientation & rotate
        # ==========================
        img = Image.open(in_path)
        img = ImageOps.exif_transpose(img) 

        if img.height > img.width:
            print(f"🔄 Portrait detected → rotating: {f}")
            img = img.rotate(90, expand=True)

        # Simpan ke file sementara
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            temp_path = tmp.name
            img.save(temp_path, quality=95)

        # ==========================
        # Start Cropping
        # ==========================
        auto_crop(temp_path, out_path)

        # Hapus file sementara
        os.remove(temp_path)

        if i % 10 == 0:
            print(f"📦 Progress: {i}/{total}")

    print(f"✅ Done. Total: {total} files.")

# ==========================
# Entry point
# ==========================
if __name__ == "__main__":
    input_dir = "input"
    output_dir = "output"

    if os.path.isdir(input_dir):
        process_dir(input_dir, output_dir)
    else:
        print("❌ Folder input tidak ditemukan.")
