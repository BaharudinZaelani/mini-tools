import json
import shutil
import logging
import pandas as pd
from pathlib import Path
from PIL import Image

# ======================
# PATH SETUP
# ======================
base_dir = Path("/content/mini-tools/dataset")
metadata_json = base_dir / "output" / "data.json"
dataset_folders = ["training", "testing", "validation"]
dimension = 1024

# ======================
# LOGGING
# ======================
LOG_DIR = base_dir / "logs"
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / "dataset_process.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ======================
# SAVE METADATA TO JSON
# ======================
metadata_file = base_dir / "metadata.xlsx"
output_json_file = f"{base_dir}/output/data.json"

if metadata_file.exists():
    df = pd.read_excel(metadata_file)
    df.to_json(
        output_json_file,
        orient="records",
        indent=2,
        force_ascii=False
    )
    print("✔ Metadata saved to JSON")
else:
    print("❌ metadata.xlsx not found")

# ======================
# LOAD METADATA
# ======================
with open(metadata_json, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# ======================
# CHECK IMAGE DIMENSION
# ======================
def checkImageDimension(path: Path) -> bool:
    with Image.open(path) as img:
        return img.size == (dimension, dimension)

# ======================
# FIND IMAGE ANYWHERE
# ======================
def find_image_anywhere(image_name: str):
    base_name = image_name.rsplit(".", 1)[0]

    for folder in dataset_folders:
        dir_path = base_dir / folder
        if not dir_path.exists():
            continue

        for file in dir_path.iterdir():
            if file.is_file():
                if file.stem == base_name:
                    return file

    return None
# ======================
# PROCESS & MOVE
# ======================
moved = 0
already_ok = 0
missing = 0

for item in metadata:
    image_url = item.get("Image_URL")
    target_type = item.get("Type")

    if not image_url or not target_type:
        continue

    filename = image_url.replace("\\", "/").split("/")[-1]
    target_dir = base_dir / target_type
    target_dir.mkdir(exist_ok=True)

    found_file = find_image_anywhere(filename)

    if not checkImageDimension(found_file):
        logger.warning(f"⚠️ SKIP SIZE | {found_file.name} | not {dimension}x{dimension}")
        continue

    if not found_file:
        logger.error(f"❌ MISSING | {filename}")
        missing += 1
        continue

    # jika sudah di folder yang benar
    if found_file.parent.name == target_type:
        logger.info(f"✔ OK | {target_type}/{found_file.name}")
        already_ok += 1
        continue

    # pindahkan file
    destination = target_dir / found_file.name

    if destination.exists():
        logger.warning(f"⚠️ SKIP (exists) | {destination}")
        continue

    shutil.move(str(found_file), str(destination))
    logger.info(f"➡️ MOVED | {found_file} → {destination}")
    moved += 1

# ======================
# SUMMARY
# ======================
logger.info("\n=== SUMMARY ===")
logger.info(f"Already OK : {already_ok}")
logger.info(f"Moved      : {moved}")
logger.info(f"Missing    : {missing}")
