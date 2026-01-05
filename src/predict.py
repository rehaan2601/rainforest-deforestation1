import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

from model import unet_model
from preprocessing import load_image

# ---------------- PATHS ----------------
MODEL_PATH = "outputs/unet_deforestation.h5"
INPUT_IMAGE_PATH = "data/sample_input.jpg"     # change this
OUTPUT_MASK_PATH = "outputs/predicted_mask.png"

IMG_SIZE = 224


def predict_mask(image_path):
    # Load and preprocess image
    image = load_image(image_path)
    image = np.expand_dims(image, axis=0)

    # Load model
    model = unet_model(input_size=(IMG_SIZE, IMG_SIZE, 3))
    model.load_weights(MODEL_PATH)

    # Predict
    prediction = model.predict(image)[0]

    # Threshold (binary mask)
    mask = (prediction > 0.5).astype(np.uint8) * 255

    return mask


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("❌ Trained model not found. Train the model first.")

    if not os.path.exists(INPUT_IMAGE_PATH):
        raise FileNotFoundError("❌ Input image not found.")

    mask = predict_mask(INPUT_IMAGE_PATH)

    # Save mask
    cv2.imwrite(OUTPUT_MASK_PATH, mask)

    print("✅ Prediction completed.")
    print(f"Predicted mask saved at: {OUTPUT_MASK_PATH}")