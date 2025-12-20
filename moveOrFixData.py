import json
import shutil
from pathlib import Path

# ======================
# PATH SETUP
# ======================
base_dir = Path("./dataset")
metadata_json = base_dir / "output" / "data.json"
dataset_folders = ["training", "testing", "validation"]

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
# FIND IMAGE ANYWHERE
# ======================
def find_image_anywhere(image_name: str):
    for folder in dataset_folders:
        dir_path = base_dir / folder
        if not dir_path.exists():
            continue

        matches = list(dir_path.glob(f"{image_name}.*"))
        if matches:
            return matches[0]  # ambil pertama
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

    image_name = Path(image_url).stem
    target_dir = base_dir / target_type
    target_dir.mkdir(exist_ok=True)

    found_file = find_image_anywhere(image_name)

    if not found_file:
        print(f"❌ MISSING | {image_name}")
        missing += 1
        continue

    # jika sudah di folder yang benar
    if found_file.parent.name == target_type:
        print(f"✔ OK | {target_type}/{found_file.name}")
        already_ok += 1
        continue

    # pindahkan file
    destination = target_dir / found_file.name

    if destination.exists():
        print(f"⚠️ SKIP (exists) | {destination}")
        continue

    shutil.move(str(found_file), str(destination))
    print(f"➡️ MOVED | {found_file} → {destination}")
    moved += 1

# ======================
# SUMMARY
# ======================
print("\n=== SUMMARY ===")
print(f"Already OK : {already_ok}")
print(f"Moved      : {moved}")
print(f"Missing    : {missing}")
