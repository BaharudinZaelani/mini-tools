import cv2
import numpy as np
import os
import tempfile
import piexif
import subprocess
from PIL import Image, ImageOps

# ==========================
# Utility
# ==========================
def calculate_solidity(contour):
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    return area / hull_area if hull_area > 0 else 0

# ==========================
# Smart detection
# ==========================
def detect_board_smart(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    lower_white = np.array([0, 0, 160])
    upper_white = np.array([180, 60, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

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

    if candidates:
        area, rect = max(candidates, key=lambda x: x[0])
        img_area = image.shape[0] * image.shape[1]
        if area < img_area * 0.1:
            return None
        return rect

    return None

def detect_chicken_direction(image, board_rect):
    if board_rect is None:
        return "center"

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    x, y, w, h = board_rect

    regions = {
        "top": edges[max(0, y - h//2):y, x:x+w],
        "bottom": edges[y+h:min(edges.shape[0], y+h+h//2), x:x+w],
        "left": edges[y:y+h, max(0, x-w//2):x],
        "right": edges[y:y+h, x+w:min(edges.shape[1], x+w+w//2)],
    }

    scores = {k: np.sum(v > 0) / (v.size + 1) for k, v in regions.items()}
    return max(scores, key=scores.get)

# ==========================
# Core crop (DIMODIFIKASI untuk pertahankan metadata)
# ==========================
def auto_crop(input_path, output_path, exif_bytes=None):
    TARGET_SIZE = 1024

    if os.path.exists(output_path):
        print(f"⏭️ Skip: {os.path.basename(output_path)}")
        return

    img = cv2.imread(input_path)
    if img is None:
        print(f"⚠️ Tidak bisa baca: {input_path}")
        return

    h_img, w_img = img.shape[:2]

    board_rect = detect_board_smart(img)
    if board_rect:
        x, y, w, h = board_rect
        direction = detect_chicken_direction(img, board_rect)

        margin_x = int(w * 0.1)
        margin_y = int(h * 0.2)
        margin_top = int(h * (0.4 if direction == "top" else 0.25))

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_top)
        x2 = min(w_img, x + w + margin_x)
        y2 = min(h_img, y + h + margin_y)

        crop = img[y1:y2, x1:x2]
    else:
        size = int(min(h_img, w_img) * 0.7)
        x1 = (w_img - size) // 2
        y1 = (h_img - size) // 2
        crop = img[y1:y1+size, x1:x1+size]

    resized = cv2.resize(crop, (TARGET_SIZE, TARGET_SIZE), cv2.INTER_LANCZOS4)

    # Konversi ke PIL Image dan simpan dengan metadata
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    
    # Simpan dengan metadata jika ada
    save_kwargs = {
        "format": "JPEG",
        "quality": 95,
        "subsampling": 0
    }
    
    if exif_bytes:
        try:
            # Update Orientation tag ke normal (1) karena sudah di-rotate
            exif_dict = piexif.load(exif_bytes)
            exif_dict["0th"][piexif.ImageIFD.Orientation] = 1
            exif_bytes = piexif.dump(exif_dict)
            save_kwargs["exif"] = exif_bytes
        except Exception as e:
            print(f"⚠️ Gagal memproses EXIF: {e}")
    
    pil_img.save(output_path, **save_kwargs)
    print(f"💾 Saved: {os.path.basename(output_path)}")

# ==========================
# Batch processing (DIPERBAIKI)
# ==========================
def process_dir(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    print(f"🧩 Processing {len(files)} files...")

    for i, f in enumerate(files, 1):
        in_path = os.path.join(input_dir, f)
        out_path = os.path.join(output_dir, f)

        # Buka file dan ekstrak metadata SEBELUM modifikasi apapun
        try:
            # Ekstrak EXIF dari file asli
            with Image.open(in_path) as img:
                # Ekstrak metadata
                exif_bytes = img.info.get("exif")
                
                # Jika tidak ada EXIF, coba ambil dengan piexif
                if exif_bytes is None:
                    try:
                        exif_bytes = piexif.load(img.info.get("exif", b""))
                        exif_bytes = piexif.dump(exif_bytes) if exif_bytes else None
                    except:
                        exif_bytes = None
                
                # Lakukan rotasi jika diperlukan (di memori saja)
                img = ImageOps.exif_transpose(img)
                
                # Rotasi 90 derajat jika tinggi > lebar
                if img.height > img.width:
                    img = img.rotate(90, expand=True, resample=Image.BICUBIC)
                
                # Simpan ke file sementara dengan metadata
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    temp_path = tmp.name
                    
                    save_kwargs = {
                        "format": "JPEG",
                        "quality": 95,
                        "subsampling": 0
                    }
                    
                    # Jika ada EXIF, update orientation ke normal
                    if exif_bytes:
                        try:
                            exif_dict = piexif.load(exif_bytes)
                            exif_dict["0th"][piexif.ImageIFD.Orientation] = 1
                            exif_bytes = piexif.dump(exif_dict)
                            save_kwargs["exif"] = exif_bytes
                        except:
                            # Jika gagal update EXIF, gunakan asli
                            save_kwargs["exif"] = exif_bytes
                    
                    img.save(temp_path, **save_kwargs)
        
        except Exception as e:
            print(f"⚠️ Error processing {f}: {e}")
            continue
        
        # Lakukan cropping dengan metadata yang sudah disiapkan
        auto_crop(temp_path, out_path, exif_bytes)
        
        # Hapus file sementara
        os.remove(temp_path)
        
        # Backup: gunakan exiftool jika metadata masih hilang
        if not exif_bytes:
            try:
                subprocess.run([
                    "exiftool",
                    "-overwrite_original",
                    "-TagsFromFile", in_path,
                    "-all:all",
                    "-Orientation=1",  # Reset orientation
                    out_path
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            except Exception as e:
                print(f"⚠️ Exiftool error untuk {f}: {e}")
        
        if i % 10 == 0:
            print(f"📦 Progress: {i}/{len(files)}")

    print("✅ Done.")

# ==========================
# Entry
# ==========================
if __name__ == "__main__":
    try:
        import piexif
    except ImportError:
        print("📦 Installing required package: piexif")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "piexif"])
        import piexif
    process_dir("input", "output")