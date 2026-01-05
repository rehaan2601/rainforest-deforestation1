import cv2
import os
import numpy as np

IMG_SIZE = 224

RAW_IMAGE_DIR = "data/raw/deforestation-dataset/images"
RAW_MASK_DIR  = "data/raw/deforestation-dataset/masks"

PROCESSED_IMAGE_DIR = "data/processed/images"
PROCESSED_MASK_DIR  = "data/processed/masks"

os.makedirs(PROCESSED_IMAGE_DIR, exist_ok=True)
os.makedirs(PROCESSED_MASK_DIR, exist_ok=True)


def preprocess_and_save():
    image_files = sorted(os.listdir(RAW_IMAGE_DIR))

    for file_name in image_files:
        image_path = os.path.join(RAW_IMAGE_DIR, file_name)
        mask_path = os.path.join(RAW_MASK_DIR, file_name)

        if not os.path.exists(mask_path):
            continue

        # --- Image ---
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = image / 255.0

        # --- Mask ---
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        mask = (mask > 0).astype(np.uint8) * 255

        # Save
        cv2.imwrite(
            os.path.join(PROCESSED_IMAGE_DIR, file_name),
            (image * 255).astype(np.uint8)
        )
        cv2.imwrite(
            os.path.join(PROCESSED_MASK_DIR, file_name),
            mask
        )

    print("✅ Preprocessing completed successfully.")


if __name__ == "__main__":
    preprocess_and_save()

    