import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from preprocess import get_generators, ALLOWED_CLASSES

SAVE_DIR = '/Users/ririputri/porto/train'
MODEL_H5 = os.path.join(SAVE_DIR, 'emotion_model_clcm.h5')

def main():
    _, val_gen, _ = get_generators()
    model = tf.keras.models.load_model(MODEL_H5)

    val_gen.reset()
    preds = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_gen.classes[:len(y_pred)]

    print("\n" + "="*60)
    print("CLASSIFICATION REPORT — CLCM FER-2013 5-class")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=ALLOWED_CLASSES))

    overall = np.mean(y_pred == y_true)
    print(f"Overall val accuracy: {overall:.4f} ({overall*100:.1f}%)")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=ALLOWED_CLASSES, yticklabels=ALLOWED_CLASSES)
    plt.title(f'Confusion Matrix — CLCM FER-2013 5-class ({overall*100:.1f}%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    out = os.path.join(SAVE_DIR, 'confusion_matrix_clcm.png')
    plt.savefig(out, dpi=150)
    print(f"\nSaved: {out}")
    plt.show()

if __name__ == '__main__':
    main()
