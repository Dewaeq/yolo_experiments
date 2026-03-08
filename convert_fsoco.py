import os
import json
import base64
import zlib
import cv2
import numpy as np
import shutil
import random
from pathlib import Path

# --- CONFIGURATION ---
CLASS_MAP = {
    "seg_blue_cone": 0,
    "seg_yellow_cone": 1,
    "seg_orange_cone": 2,
    "seg_large_orange_cone": 3,
    "seg_unknown_cone": 4,
}
SPLIT_RATIO = 0.8  # 80% Training, 20% Validation
RANDOM_SEED = 42  # Ensures the split is the same if you re-run the script


def decode_bitmap(b64_string):
    """Decodes the Supervisely base64 zlib string into a binary mask."""
    z = zlib.decompress(base64.b64decode(b64_string))
    n = np.frombuffer(z, np.uint8)
    mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)
    # Check if the image has multiple channels (e.g., RGBA or RGB)
    if len(mask.shape) == 3:
        if mask.shape[2] == 4:
            # It's RGBA. The actual mask is stored in the Alpha channel.
            mask = mask[:, :, 3]
        else:
            # It's RGB. Convert to grayscale.
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Ensure it's a clean binary image (pixels are either 0 or 255)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    return mask


def process_fsoco_dataset(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir).absolute()  # Absolute path needed for YOLO yaml

    # 1. Setup YOLO output directories
    dirs_to_make = [
        output_path / "images" / "train",
        output_path / "images" / "val",
        output_path / "labels" / "train",
        output_path / "labels" / "val",
    ]
    for d in dirs_to_make:
        d.mkdir(parents=True, exist_ok=True)

    valid_samples = []  # Will hold tuples of (src_img_path, list_of_yolo_lines)

    print(f"Scanning FSOCO directory: {input_path}...")

    # 2. Iterate through all subfolders
    for subfolder in input_path.iterdir():
        if not subfolder.is_dir():
            continue

        ann_dir = subfolder / "ann"
        img_dir = subfolder / "img"

        # Skip folders that don't have the expected Supervisely structure
        if not ann_dir.exists() or not img_dir.exists():
            continue

        print(f"  Processing subfolder: {subfolder.name}")

        # 3. Process each JSON annotation
        for json_file in ann_dir.glob("*.json"):
            # Supervisely usually names annotations like "image.png.json"
            img_filename = json_file.name.replace(".json", "")
            src_img_path = img_dir / img_filename

            if not src_img_path.exists():
                continue  # Skip if the image is missing

            with open(json_file, "r") as f:
                data = json.load(f)

            img_width = data["size"]["width"]
            img_height = data["size"]["height"]

            yolo_lines = []

            for obj in data.get("objects", []):
                class_name = obj["classTitle"]

                # We only care about segmentation masks
                if class_name not in CLASS_MAP or "bitmap" not in obj:
                    continue

                class_id = CLASS_MAP[class_name]

                # Decode the mask and its offset
                mask = decode_bitmap(obj["bitmap"]["data"])
                origin_x, origin_y = obj["bitmap"]["origin"]

                # Extract the outline (contour) of the mask
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if not contours:
                    continue

                # Grab the largest contour to avoid stray pixels
                contour = max(contours, key=cv2.contourArea)

                # Convert to normalized YOLO coordinates
                polygon = []
                for point in contour:
                    x = (point[0][0] + origin_x) / img_width
                    y = (point[0][1] + origin_y) / img_height

                    # Clamp between 0.0 and 1.0
                    x = max(0.0, min(1.0, x))
                    y = max(0.0, min(1.0, y))

                    polygon.append(f"{x:.6f} {y:.6f}")

                yolo_lines.append(f"{class_id} " + " ".join(polygon))

            # If we found at least one valid cone mask in this image, keep it
            if yolo_lines:
                valid_samples.append((src_img_path, yolo_lines))

    print(f"\nFound {len(valid_samples)} valid images with segmentation masks.")

    # 4. Shuffle and Split (Train / Val)
    random.seed(RANDOM_SEED)
    random.shuffle(valid_samples)

    split_index = int(len(valid_samples) * SPLIT_RATIO)
    train_samples = valid_samples[:split_index]
    val_samples = valid_samples[split_index:]

    print(f"Splitting data: {len(train_samples)} Train | {len(val_samples)} Validation")

    # 5. Write Files Function
    def write_dataset(samples, split_name):
        for img_path, yolo_lines in samples:
            # Create a clean unique filename (in case different subfolders have identically named files)
            safe_filename = f"{img_path.parent.parent.name}_{img_path.name}"
            safe_txtname = safe_filename.rsplit(".", 1)[0] + ".txt"

            # Copy image
            dst_img = output_path / "images" / split_name / safe_filename
            shutil.copy(img_path, dst_img)

            # Write labels
            dst_txt = output_path / "labels" / split_name / safe_txtname
            with open(dst_txt, "w") as f:
                f.write("\n".join(yolo_lines))

    print("Copying Training files...")
    write_dataset(train_samples, "train")
    print("Copying Validation files...")
    write_dataset(val_samples, "val")

    # 6. Generate dataset.yaml automatically
    yaml_path = output_path / "fsoco.yaml"
    yaml_content = f"""path: {output_path}
train: images/train
val: images/val

names:
  0: blue_cone
  1: yellow_cone
  2: orange_cone
  3: large_orange_cone
  4: unknown_cone
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"\nSuccess! YOLO dataset and YAML config created at:\n{output_path}")
    print(f"You can now train using: model.train(data='{yaml_path}', epochs=50)")


# --- RUN THE PIPELINE ---
# 1. Point this to the unzipped FSOCO dataset folder (the one containing meta.json)
INPUT_DATASET = "./fsoco_segmentation_train"
# 2. Where you want the YOLO formatted dataset to be generated
OUTPUT_DATASET = "./yolo_fsoco"

if __name__ == "__main__":
    process_fsoco_dataset(INPUT_DATASET, OUTPUT_DATASET)
