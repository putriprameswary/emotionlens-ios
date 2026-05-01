import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from preprocess import get_generators
from model import build_clcm_model

SAVE_DIR = '/Users/ririputri/porto/train'
MODEL_H5 = os.path.join(SAVE_DIR, 'emotion_model_clcm.h5')

def get_callbacks():
    return [
        EarlyStopping(monitor='val_accuracy', patience=7,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=3, min_lr=1e-7, verbose=1)
    ]

def main():
    print("Loading generators...")
    train_gen, val_gen, class_weight_dict = get_generators()

    # PHASE 1 — frozen backbone
    print("\n[PHASE 1] Frozen backbone...")
    model = build_clcm_model()
    model.compile(optimizer=Adam(1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    h1 = model.fit(train_gen, validation_data=val_gen,
                   epochs=25, class_weight=class_weight_dict,
                   callbacks=get_callbacks(), verbose=1)
    print(f"Phase 1 best val_accuracy: {max(h1.history['val_accuracy']):.4f}")

    # PHASE 2 — unfreeze semua layer model
    print("\n[PHASE 2] Full fine-tuning...")
    
    # Unfreeze SEMUA layer di seluruh model (tidak perlu get_layer)
    for layer in model.layers:
        layer.trainable = True

    trainable = sum(1 for l in model.layers if l.trainable)
    print(f"Trainable layers: {trainable}")

    model.compile(optimizer=Adam(1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    h2 = model.fit(train_gen, validation_data=val_gen,
                   epochs=30, class_weight=class_weight_dict,
                   callbacks=get_callbacks(), verbose=1)
    p2_best = max(h2.history['val_accuracy'])
    print(f"Phase 2 best val_accuracy: {p2_best:.4f}")

    model.save(MODEL_H5)
    print(f"Saved: {MODEL_H5}")
    print(f"Input name: {model.input_names}")

    import pickle
    with open(os.path.join(SAVE_DIR, 'history.pkl'), 'wb') as f:
        pickle.dump({'h1': h1.history, 'h2': h2.history}, f)

if __name__ == '__main__':
    main()
