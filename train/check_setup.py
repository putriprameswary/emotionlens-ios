# Simpan sebagai: check_setup.py
# Jalankan: python check_setup.py

import tensorflow as tf
import numpy as np

print("="*50)
print(f"TensorFlow version : {tf.__version__}")
print(f"GPU devices        : {tf.config.list_physical_devices('GPU')}")
print(f"Metal available    : {'YES ✅' if tf.config.list_physical_devices('GPU') else 'NO ❌ (CPU only)'}")
print("="*50)

# Cek dataset bisa dibaca
import os
BASE = '/Users/ririputri/porto/images'
CLASSES = ['angry', 'happy', 'neutral', 'sad', 'surprise']

print("\nDataset check:")
for split in ['train', 'validation']:
    for cls in CLASSES:
        path = os.path.join(BASE, split, cls)
        if os.path.exists(path):
            count = len(os.listdir(path))
            print(f"  ✅ {split}/{cls}: {count} files")
        else:
            print(f"  ❌ MISSING: {split}/{cls}")

# Cek disgust/fear tidak ikut terbaca
print("\nChecking excluded classes:")
for split in ['train', 'validation']:
    for cls in ['disgust', 'fear']:
        path = os.path.join(BASE, split, cls)
        if os.path.exists(path):
            print(f"  ⚠️  {split}/{cls} EXISTS on disk — will be IGNORED by whitelist ✅")
        else:
            print(f"  ✅ {split}/{cls} not found — clean")

print("\nSetup OK! Ready to train.")