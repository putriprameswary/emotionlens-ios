import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

ALLOWED_CLASSES = ['angry', 'happy', 'neutral', 'sad', 'surprise']
TRAIN_DIR = '/Users/ririputri/porto/images/train'
VAL_DIR   = '/Users/ririputri/porto/images/validation'
IMG_SIZE  = (96, 96)
BATCH_SIZE = 32

def get_generators():
    # MobileNetV2 expects input [0,1] — rescale=1/255 is correct
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=15,
        zoom_range=0.10,
        width_shift_range=0.10,
        height_shift_range=0.10,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        classes=ALLOWED_CLASSES,
        shuffle=True
    )
    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        classes=ALLOWED_CLASSES,
        shuffle=False
    )

    assert train_gen.class_indices == {c: i for i, c in enumerate(ALLOWED_CLASSES)}, \
        f"Class mismatch: {train_gen.class_indices}"

    labels = train_gen.classes
    cw = compute_class_weight('balanced', classes=np.arange(5), y=labels)
    class_weight_dict = {i: float(w) for i, w in enumerate(cw)}

    print(f"Class indices : {train_gen.class_indices}")
    print(f"Train samples : {train_gen.samples}")
    print(f"Val samples   : {val_gen.samples}")
    print(f"Class weights : {class_weight_dict}")
    return train_gen, val_gen, class_weight_dict
