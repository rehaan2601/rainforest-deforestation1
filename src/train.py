import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from preprocessing import load_dataset
from model import unet_model


# -------- Paths (update when dataset is ready) --------
IMAGE_DIR = "data/processed/images"
MASK_DIR = "data/processed/masks"

# -------- Hyperparameters --------
BATCH_SIZE = 8
EPOCHS = 20
IMG_SIZE = 224


def train():
    print("Loading dataset...")
    images, masks = load_dataset(IMAGE_DIR, MASK_DIR)

    masks = np.expand_dims(masks, axis=-1)

    X_train, X_val, y_train, y_val = train_test_split(
        images, masks, test_size=0.2, random_state=42
    )

    print("Building model...")
    model = unet_model(input_size=(IMG_SIZE, IMG_SIZE, 3))
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        ModelCheckpoint(
            "outputs/unet_deforestation.h5",
            save_best_only=True,
            monitor="val_loss"
        ),
        EarlyStopping(
            patience=5,
            restore_best_weights=True
        )
    ]

    print("Training started...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )

    print("Training completed and model saved.")


if __name__ == "__main__":
    train()
