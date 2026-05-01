import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from preprocess import get_generators, ALLOWED_CLASSES

SAVE_DIR  = '/Users/ririputri/porto/train'
MODEL_IN  = os.path.join(SAVE_DIR, 'emotion_model_clcm.h5')
MODEL_OUT = os.path.join(SAVE_DIR, 'emotion_model_clcm_v2.h5')

def get_callbacks():
    return [
        EarlyStopping(monitor='val_accuracy', patience=8,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=3, min_lr=1e-8, verbose=1)
    ]

def main():
    print("Loading generators...")
    train_gen, val_gen, _ = get_generators()

    # Hitung class weight yang lebih agresif untuk angry dan sad
    # angry=0, happy=1, neutral=2, sad=3, surprise=4
    class_weight_aggressive = {
        0: 2.5,   # angry — boost signifikan
        1: 0.6,   # happy — kurangi dominasi
        2: 0.9,   # neutral
        3: 2.0,   # sad — boost
        4: 1.5    # surprise
    }
    print(f"Aggressive class weights: {class_weight_aggressive}")

    # Load model terbaik sebelumnya
    print(f"\nLoading best model: {MODEL_IN}")
    model = tf.keras.models.load_model(MODEL_IN)
    print(f"Loaded. Input name: {model.input_names}")

    # Semua layer trainable
    for layer in model.layers:
        layer.trainable = True

    # LR sangat kecil — fine-tune halus dari checkpoint 64.8%
    model.compile(optimizer=Adam(2e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print(f"\n[PHASE 3] Fine-tuning dari checkpoint 64.8%...")
    print("Target: val_accuracy >= 0.70")

    h3 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=30,
        class_weight=class_weight_aggressive,
        callbacks=get_callbacks(),
        verbose=1
    )

    p3_best = max(h3.history['val_accuracy'])
    print(f"\nPhase 3 best val_accuracy: {p3_best:.4f} ({p3_best*100:.1f}%)")

    if p3_best >= 0.70:
        print("🎉 Target 70% tercapai!")
    else:
        print(f"Belum 70% — gap: {(0.70 - p3_best)*100:.1f}%")

    model.save(MODEL_OUT)
    print(f"Saved: {MODEL_OUT}")

if __name__ == '__main__':
    main()
