"""
evaluate_v3.py — Evaluasi detail untuk model v3
================================================
Menampilkan:
  - Classification report lengkap
  - Confusion matrix dengan persen per baris
  - Analisis misclassification angry/sad vs neutral
  - Perbandingan v1 (64.8%) vs v3

Cara pakai:
    python3 evaluate_v3.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from preprocess import get_generators, ALLOWED_CLASSES

SAVE_DIR  = '/Users/ririputri/porto/train'
MODEL_V3  = os.path.join(SAVE_DIR, 'emotion_model_clcm_v3.h5')
MODEL_V1  = os.path.join(SAVE_DIR, 'emotion_model_clcm.h5')

# Baseline dari training sebelumnya (untuk perbandingan)
BASELINE = {
    'angry'   : {'precision': 0.69, 'recall': 0.43, 'f1': 0.53},
    'happy'   : {'precision': 0.87, 'recall': 0.72, 'f1': 0.79},
    'neutral' : {'precision': 0.49, 'recall': 0.72, 'f1': 0.58},
    'sad'     : {'precision': 0.52, 'recall': 0.56, 'f1': 0.54},
    'surprise': {'precision': 0.78, 'recall': 0.79, 'f1': 0.78},
}
BASELINE_ACC = 0.648


def evaluate_model(model, val_gen, label):
    val_gen.reset()
    preds  = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_gen.classes[:len(y_pred)]
    return y_true, y_pred, preds


def plot_confusion_matrix_pct(cm, classes, title, ax):
    """Confusion matrix dengan persentase per baris (recall per kelas)."""
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(
        cm_pct, annot=True, fmt='.1f', cmap='Blues',
        xticklabels=classes, yticklabels=classes,
        ax=ax, vmin=0, vmax=100,
        annot_kws={'size': 10}
    )
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_ylabel('True Label', fontsize=10)
    ax.set_xlabel('Predicted Label', fontsize=10)

    # Tambah nilai absolut di dalam cell
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j + 0.5, i + 0.72,
                    f'({cm[i, j]})',
                    ha='center', va='center',
                    fontsize=7, color='gray')


def analyze_hard_classes(y_true, y_pred, preds, classes):
    """Analisis misprediksi untuk angry (0) dan sad (3)."""
    print("\n" + "─" * 60)
    print("ANALISIS HARD CLASSES (angry & sad)")
    print("─" * 60)

    for cls_idx, cls_name in [(0, 'angry'), (3, 'sad')]:
        mask_true = y_true == cls_idx
        total = mask_true.sum()
        if total == 0:
            continue

        pred_for_cls = y_pred[mask_true]
        conf_for_cls = preds[mask_true]

        correct = (pred_for_cls == cls_idx).sum()
        recall  = correct / total

        print(f"\n{cls_name.upper()} (total: {total})")
        print(f"  Recall: {recall:.3f} ({correct}/{total} benar)")
        print(f"  Misprediksi breakdown:")

        for wrong_cls in range(len(classes)):
            if wrong_cls == cls_idx:
                continue
            wrong_count = (pred_for_cls == wrong_cls).sum()
            if wrong_count > 0:
                pct = wrong_count / total * 100
                # Cek confidence rata-rata untuk misprediksi ini
                wrong_mask = pred_for_cls == wrong_cls
                avg_conf = conf_for_cls[wrong_mask, wrong_cls].mean()
                print(f"    → prediksi {classes[wrong_cls]:8s}: {wrong_count:3d} ({pct:.1f}%) | avg conf: {avg_conf:.3f}")


def print_comparison_table(y_true, y_pred, classes):
    """Tabel perbandingan v1 vs v3."""
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    new_acc = np.mean(y_pred == y_true)

    print("\n" + "=" * 70)
    print("PERBANDINGAN: v1 (64.8%) vs v3")
    print("=" * 70)
    print(f"{'Kelas':<10} {'Recall v1':>10} {'Recall v3':>10} {'Delta':>8} {'F1 v1':>8} {'F1 v3':>8} {'Delta':>8}")
    print("─" * 70)

    for cls in classes:
        r_old = BASELINE[cls]['recall']
        f_old = BASELINE[cls]['f1']
        r_new = report[cls]['recall']
        f_new = report[cls]['f1-score']

        r_delta = r_new - r_old
        f_delta = f_new - f_old

        r_sign = '+' if r_delta >= 0 else ''
        f_sign = '+' if f_delta >= 0 else ''

        print(f"{cls:<10} {r_old:>10.3f} {r_new:>10.3f} {r_sign+f'{r_delta:.3f}':>8} "
              f"{f_old:>8.3f} {f_new:>8.3f} {f_sign+f'{f_delta:.3f}':>8}")

    print("─" * 70)
    acc_delta = new_acc - BASELINE_ACC
    acc_sign  = '+' if acc_delta >= 0 else ''
    print(f"{'Accuracy':<10} {BASELINE_ACC:>10.3f} {new_acc:>10.3f} {acc_sign+f'{acc_delta:.3f}':>8}")
    print("=" * 70)


def main():
    print("=" * 60)
    print("evaluate_v3.py — Model v3 Evaluation")
    print("=" * 60)

    _, val_gen, _ = get_generators()

    # Load model v3
    if not os.path.exists(MODEL_V3):
        print(f"Model v3 tidak ditemukan: {MODEL_V3}")
        print("Jalankan train_v3.py terlebih dahulu.")
        return

    print(f"\nLoading model v3: {MODEL_V3}")
    model_v3 = tf.keras.models.load_model(
        MODEL_V3,
        custom_objects={'loss_fn': lambda yt, yp: yp}  # placeholder
    )

    print("\nEvaluating model v3...")
    y_true, y_pred, preds = evaluate_model(model_v3, val_gen, 'v3')

    # Classification report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT — v3")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=ALLOWED_CLASSES))

    overall = np.mean(y_pred == y_true)
    print(f"Overall val accuracy: {overall:.4f} ({overall*100:.1f}%)")

    if overall >= 0.70:
        print("Target 70% TERCAPAI!")
    else:
        print(f"Belum 70% — gap: {(0.70 - overall)*100:.1f}%")

    # Analisis per kelas
    analyze_hard_classes(y_true, y_pred, preds, ALLOWED_CLASSES)

    # Tabel perbandingan
    print_comparison_table(y_true, y_pred, ALLOWED_CLASSES)

    # Plot
    fig = plt.figure(figsize=(14, 6))
    gs  = gridspec.GridSpec(1, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    cm_v3 = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix_pct(cm_v3, ALLOWED_CLASSES,
                              f'v3 — {overall*100:.1f}% accuracy\n(% = recall per baris, (n) = count)',
                              ax1)

    # Plot recall comparison bar chart
    ax2 = fig.add_subplot(gs[0, 1])
    report = classification_report(y_true, y_pred, target_names=ALLOWED_CLASSES, output_dict=True)
    recalls_v1 = [BASELINE[c]['recall'] for c in ALLOWED_CLASSES]
    recalls_v3 = [report[c]['recall'] for c in ALLOWED_CLASSES]
    x = np.arange(len(ALLOWED_CLASSES))
    width = 0.35

    bars1 = ax2.bar(x - width/2, recalls_v1, width, label='v1 (64.8%)', color='#aaa', alpha=0.8)
    bars2 = ax2.bar(x + width/2, recalls_v3, width, label=f'v3 ({overall*100:.1f}%)', color='#3B82F6', alpha=0.9)

    ax2.axhline(y=0.7, color='red', linestyle='--', linewidth=1, label='Target 70%')
    ax2.set_xlabel('Kelas')
    ax2.set_ylabel('Recall')
    ax2.set_title('Recall per kelas: v1 vs v3')
    ax2.set_xticks(x)
    ax2.set_xticklabels(ALLOWED_CLASSES)
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords='offset points',
                     ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    out = os.path.join(SAVE_DIR, 'evaluation_v3_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nPlot disimpan: {out}")
    plt.show()


if __name__ == '__main__':
    main()