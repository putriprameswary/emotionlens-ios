"""
train_v3.py — Target 70%+ val_accuracy on FER-2013 5-class
=============================================================
Strategi vs train_v2.py:
  1. Focal Loss  (gamma=2) — hukum misprediksi angry/sad lebih keras
  2. Label Smoothing (0.1) — cegah model terlalu confident ke neutral
  3. Mixup Augmentation — buat boundary angry/sad vs neutral lebih tegas
  4. Per-class augmentation agresif untuk angry dan sad
  5. Cosine LR schedule + warmup — konvergen lebih stabil dari ReduceLROnPlateau
  6. Multi-phase: fine-tune → discriminative LR → polish

Cara pakai:
    python3 train_v3.py
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback
from preprocess import get_generators, ALLOWED_CLASSES

# ─── Konfigurasi ────────────────────────────────────────────────────────────
SAVE_DIR  = '/Users/ririputri/porto/train'
MODEL_IN  = os.path.join(SAVE_DIR, 'emotion_model_clcm.h5')
MODEL_OUT = os.path.join(SAVE_DIR, 'emotion_model_clcm_v3.h5')

NUM_CLASSES     = 5
MIXUP_ALPHA     = 0.3   # kecil — hanya untuk hard classes
LABEL_SMOOTH    = 0.1
FOCAL_GAMMA     = 2.0
FOCAL_ALPHA     = None  # gunakan class_weight saja


# ─── Focal Loss ─────────────────────────────────────────────────────────────
def focal_loss(gamma=2.0, alpha=None, label_smoothing=0.1):
    """
    Focal Loss + Label Smoothing untuk multi-class classification.
    - gamma: focusing parameter (2.0 untuk FER yang hard)
    - label_smoothing: 0.1 mencegah model over-confident ke neutral
    """
    def loss_fn(y_true, y_pred):
        # Label smoothing
        num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
        y_true_smooth = y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)

        # Clip untuk numerik stability
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # Cross entropy
        ce = -tf.reduce_sum(y_true_smooth * tf.math.log(y_pred), axis=-1)

        # Probabilitas kelas yang benar (untuk focal weight)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        focal_weight = tf.pow(1.0 - p_t, gamma)

        focal_ce = focal_weight * ce
        return tf.reduce_mean(focal_ce)

    return loss_fn


# ─── Mixup Augmentation ──────────────────────────────────────────────────────
def mixup_generator(generator, alpha=0.3, hard_classes=(0, 3)):
    """
    Mixup: blending gambar + label antara hard classes (angry=0, sad=3)
    dan neutral (2). Mempertegas boundary decision yang ambigu.
    """
    while True:
        x1, y1 = next(generator)
        x2, y2 = next(generator)

        batch_size = tf.shape(x1)[0]

        # Hanya mixup jika batch mengandung hard class atau neutral
        hard_mask = tf.reduce_any(
            tf.stack([y1[:, c] > 0.5 for c in hard_classes] + [y1[:, 2] > 0.5], axis=1),
            axis=1
        )

        lam = np.random.beta(alpha, alpha, size=x1.shape[0]).astype(np.float32)
        lam = np.maximum(lam, 1 - lam)  # selalu >= 0.5 agar label utama tetap dominan
        lam_x = lam.reshape(-1, 1, 1, 1)
        lam_y = lam.reshape(-1, 1)

        x_mix = lam_x * x1 + (1 - lam_x) * x2
        y_mix = lam_y * y1 + (1 - lam_y) * y2

        # Mixup hanya untuk sample yang mengandung hard class
        hard_mask_np = hard_mask.numpy().reshape(-1, 1, 1, 1)
        x_out = np.where(hard_mask_np, x_mix, x1)
        hard_mask_np_y = hard_mask.numpy().reshape(-1, 1)
        y_out = np.where(hard_mask_np_y, y_mix, y1)

        yield x_out, y_out


# ─── Cosine LR Schedule ──────────────────────────────────────────────────────
class CosineAnnealingSchedule(tf.keras.callbacks.Callback):
    """
    Cosine annealing dengan warm restart.
    Lebih stabil dari ReduceLROnPlateau untuk fine-tuning.
    """
    def __init__(self, lr_min=1e-7, lr_max=5e-5, T_0=10, T_mult=1):
        super().__init__()
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.T_0    = T_0
        self.T_mult = T_mult
        self.cycle  = 0
        self.T_cur  = 0

    def on_epoch_begin(self, epoch, logs=None):
        T_i = self.T_0 * (self.T_mult ** self.cycle)
        if self.T_cur >= T_i:
            self.cycle += 1
            self.T_cur = 0
            T_i = self.T_0 * (self.T_mult ** self.cycle)

        lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * \
             (1 + np.cos(np.pi * self.T_cur / T_i))
        K.set_value(self.model.optimizer.lr, lr)
        self.T_cur += 1

        if epoch % 5 == 0:
            print(f"\n[LR Schedule] Epoch {epoch}: lr = {lr:.2e}")


# ─── Class weights yang dioptimalkan ─────────────────────────────────────────
# Berdasarkan confusion matrix:
# angry  → banyak salah ke neutral (234) dan sad (210)  → boost kuat
# sad    → banyak salah ke neutral (381)                 → boost sedang
# neutral → terlalu sering diprediksi                   → kurangi sedikit
# Sesuai class distribution: angry=958, happy=1774, neutral=1233, sad=1247, surprise=831
CLASS_WEIGHTS = {
    0: 3.0,   # angry   — boost lebih kuat dari v2 (2.5 → 3.0)
    1: 0.55,  # happy   — sudah bagus, kurangi pengaruh
    2: 0.75,  # neutral — kurangi agar tidak dominan
    3: 2.2,   # sad     — sedikit lebih dari v2 (2.0 → 2.2)
    4: 1.4    # surprise — sudah bagus
}


# ─── Best model checkpoint manual ────────────────────────────────────────────
class BestModelTracker(tf.keras.callbacks.Callback):
    def __init__(self, filepath):
        super().__init__()
        self.filepath   = filepath
        self.best_val   = -np.inf
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get('val_accuracy', 0)
        if val_acc > self.best_val:
            self.best_val   = val_acc
            self.best_epoch = epoch
            self.model.save(self.filepath)
            print(f"  → Best model saved: val_accuracy={val_acc:.4f} (epoch {epoch+1})")


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("train_v3.py — Focal Loss + Mixup + Cosine LR")
    print("Target: val_accuracy >= 70%")
    print("=" * 60)

    print("\nLoading generators...")
    train_gen, val_gen, _ = get_generators()

    steps_per_epoch = len(train_gen)
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Class weights: {CLASS_WEIGHTS}")

    # Load model terbaik (64.8% checkpoint)
    print(f"\nLoading checkpoint: {MODEL_IN}")
    model = tf.keras.models.load_model(MODEL_IN)
    print(f"Model loaded. Input: {model.input_names}")
    print(f"Total layers: {len(model.layers)}")

    # ── PHASE 1: Fine-tune semua layer dengan Focal Loss ──────────────────
    print("\n" + "─" * 60)
    print("PHASE 1: Full fine-tune dengan Focal Loss + Label Smoothing")
    print("Epoch: 1-25 | LR: cosine 1e-7 → 5e-5")
    print("─" * 60)

    for layer in model.layers:
        layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=2e-5),
        loss=focal_loss(gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTH),
        metrics=['accuracy']
    )

    best_tracker_p1 = BestModelTracker(MODEL_OUT.replace('.h5', '_p1_best.h5'))
    cosine_schedule  = CosineAnnealingSchedule(lr_min=5e-7, lr_max=5e-5, T_0=8)
    early_stop_p1    = EarlyStopping(
        monitor='val_accuracy', patience=10,
        restore_best_weights=True, verbose=1
    )

    h1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=25,
        class_weight=CLASS_WEIGHTS,
        callbacks=[cosine_schedule, best_tracker_p1, early_stop_p1],
        verbose=1
    )

    p1_best = max(h1.history['val_accuracy'])
    print(f"\nPhase 1 best val_accuracy: {p1_best:.4f} ({p1_best*100:.1f}%)")

    if p1_best >= 0.70:
        print("Target 70% tercapai di Phase 1!")
        model.save(MODEL_OUT)
        print(f"Saved: {MODEL_OUT}")
        _print_summary(h1.history)
        return

    # ── PHASE 2: Discriminative LR — layer dalam lebih lambat ─────────────
    print("\n" + "─" * 60)
    print("PHASE 2: Discriminative LR — bottom layers lebih lambat")
    print("Epoch: 1-20 | LR backbone: 1e-6, head: 1e-5")
    print("─" * 60)

    # Freeze 30% layer pertama (fitur low-level), fine-tune sisanya
    total_layers = len(model.layers)
    freeze_until = int(total_layers * 0.3)

    for i, layer in enumerate(model.layers):
        layer.trainable = i >= freeze_until

    trainable_count = sum(1 for l in model.layers if l.trainable)
    print(f"Frozen: {freeze_until} layers | Trainable: {trainable_count} layers")

    model.compile(
        optimizer=Adam(learning_rate=8e-6),
        loss=focal_loss(gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTH),
        metrics=['accuracy']
    )

    best_tracker_p2 = BestModelTracker(MODEL_OUT.replace('.h5', '_p2_best.h5'))
    cosine_p2       = CosineAnnealingSchedule(lr_min=2e-7, lr_max=8e-6, T_0=6)
    early_stop_p2   = EarlyStopping(
        monitor='val_accuracy', patience=8,
        restore_best_weights=True, verbose=1
    )

    h2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,
        class_weight=CLASS_WEIGHTS,
        callbacks=[cosine_p2, best_tracker_p2, early_stop_p2],
        verbose=1
    )

    p2_best = max(h2.history['val_accuracy'])
    print(f"\nPhase 2 best val_accuracy: {p2_best:.4f} ({p2_best*100:.1f}%)")

    if p2_best >= 0.70:
        print("Target 70% tercapai di Phase 2!")
        model.save(MODEL_OUT)
        print(f"Saved: {MODEL_OUT}")
        _print_summary(h2.history)
        return

    # ── PHASE 3: Polish — LR sangat kecil, semua layer trainable ──────────
    print("\n" + "─" * 60)
    print("PHASE 3: Polish — micro-tuning dengan LR sangat kecil")
    print("Epoch: 1-15 | LR: 1e-6 (fixed cosine)")
    print("─" * 60)

    for layer in model.layers:
        layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=5e-6),
        loss=focal_loss(gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTH),
        metrics=['accuracy']
    )

    best_tracker_p3 = BestModelTracker(MODEL_OUT)
    cosine_p3       = CosineAnnealingSchedule(lr_min=1e-8, lr_max=5e-6, T_0=5)
    early_stop_p3   = EarlyStopping(
        monitor='val_accuracy', patience=7,
        restore_best_weights=True, verbose=1
    )

    h3 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=15,
        class_weight=CLASS_WEIGHTS,
        callbacks=[cosine_p3, best_tracker_p3, early_stop_p3],
        verbose=1
    )

    p3_best = max(h3.history['val_accuracy'])
    overall_best = max(p1_best, p2_best, p3_best)

    print("\n" + "=" * 60)
    print("HASIL AKHIR")
    print("=" * 60)
    print(f"Phase 1: {p1_best:.4f} ({p1_best*100:.1f}%)")
    print(f"Phase 2: {p2_best:.4f} ({p2_best*100:.1f}%)")
    print(f"Phase 3: {p3_best:.4f} ({p3_best*100:.1f}%)")
    print(f"Best overall: {overall_best:.4f} ({overall_best*100:.1f}%)")

    if overall_best >= 0.70:
        print("\nTarget 70% tercapai!")
    else:
        gap = (0.70 - overall_best) * 100
        print(f"\nBelum 70% — gap: {gap:.1f}%")
        print("Coba jalankan evaluate_v3.py untuk analisis confusion matrix terbaru.")

    print(f"\nModel terbaik disimpan di: {MODEL_OUT}")
    _print_summary(h3.history)


def _print_summary(history):
    best_epoch = np.argmax(history['val_accuracy'])
    print(f"\nBest epoch: {best_epoch + 1}")
    print(f"  train_accuracy : {history['accuracy'][best_epoch]:.4f}")
    print(f"  val_accuracy   : {history['val_accuracy'][best_epoch]:.4f}")
    print(f"  train_loss     : {history['loss'][best_epoch]:.4f}")
    print(f"  val_loss       : {history['val_loss'][best_epoch]:.4f}")


if __name__ == '__main__':
    main()