# 📱🙋🏻‍♀️ EmotionLens

> *Let your iPhone understand how you feel — in real time.*

<div align="center">

![iOS](https://img.shields.io/badge/iOS-16+-000000?style=for-the-badge\&logo=apple\&logoColor=white)
![Swift](https://img.shields.io/badge/Swift-5.0+-FA7343?style=for-the-badge\&logo=swift\&logoColor=white)
![CoreML](https://img.shields.io/badge/CoreML-On--Device-FF6F00?style=for-the-badge)
![Vision](https://img.shields.io/badge/Vision-Face%20Detection-0A84FF?style=for-the-badge)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Training-FF6F00?style=for-the-badge\&logo=tensorflow\&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-iPhone-black?style=for-the-badge\&logo=apple)
![Status](https://img.shields.io/badge/Status-Production%20Ready-34C759?style=for-the-badge)
![AI](https://img.shields.io/badge/AI-On--Device-blueviolet?style=for-the-badge)

**A real-time facial emotion recognition iOS app — fully on-device, powered by deep learning.**

[📄 Full Technical Doc](./TECHNICAL_DOC.md) · [🧠 Training Code](./train) · [📦 Dataset (FER-2013)](https://www.kaggle.com/datasets/msambare/fer2013)

</div>

---

## ✨ What Is This?

**EmotionLens** is an end-to-end AI project — from training a deep learning model to deploying it into a real-time iOS application.

Point your camera at a face, and the app instantly predicts:

* 😊 **Happy**
* 😐 **Neutral**
* 😢 **Sad**
* 😠 **Angry**
* 😲 **Surprise**

⚡ Real-time
🔒 Fully offline (on-device)
📱 Runs directly on iPhone

---

## 🧠 How It Works

```
Camera → Face Detection → Emotion Model → Smoothing → UI Overlay
```

1. Face detection using Vision Framework
2. Crop & resize to 96×96
3. Core ML inference
4. Temporal smoothing (reduce flicker)
5. UI overlay (emoji + confidence bars)

---

## 🏗 End-to-End Pipeline

```
FER-2013 Dataset
   ↓
Preprocessing + Augmentation
   ↓
CLCM Model (MobileNetV2 α=0.75)
   ↓
TensorFlow (.h5)
   ↓
CoreMLTools
   ↓
EmotionClassifier.mlpackage
   ↓
iOS App (SwiftUI + Vision + Core ML)
```

---

## 📊 Model Performance

| Metric    | Value              |
| --------- | ------------------ |
| Accuracy  | **65.3%**          |
| Model     | MobileNetV2 (CLCM) |
| Classes   | 5 emotions         |
| Inference | < 50ms             |
| Platform  | iOS (on-device)    |

---

## ⚙️ Tech Stack

| Layer      | Tools                                  |
| ---------- | -------------------------------------- |
| iOS        | SwiftUI, Vision, AVFoundation, Core ML |
| ML         | TensorFlow, Keras                      |
| Conversion | CoreMLTools                            |
| Dataset    | FER-2013                               |

---

## 🎯 Features

* 📷 Real-time face detection
* 👥 Multi-face support
* 😊 Emotion classification per face
* 📊 Confidence bars visualization
* ⚡ Smooth predictions (no flickering)
* 🔒 Fully on-device (privacy-friendly)

---

## 📂 Project Structure

```
EmotionLens/
├── Camera/
├── Vision/
├── ML/
├── UI/

train/

EmotionClassifier.mlpackage
```

---

## 🔬 Model Training

* Backbone: MobileNetV2 (α=0.75)
* Input: 96×96 RGB
* Dataset: FER-2013 (5 classes)
* Augmentation: flip, rotation, zoom, brightness
* Training:

  * Phase 1: frozen backbone
  * Phase 2: full fine-tuning

👉 **[View Full Training & Analysis →](./TECHNICAL_DOC.md)**

---

## ▶️ Run Locally

```bash
# 1. Clone repository
git clone https://github.com/putriprameswary/emotionlens-ios.git
cd emotionlens-ios

# 2. Open in Xcode
open EmotionLens.xcodeproj

# 3. Run on iPhone (recommended)
```

> ⚠️ Make sure camera permission is enabled.

---

## ⚠️ Limitations

* Grayscale dataset (FER-2013)
* Angry & Sad often confused with Neutral
* Surprise class underrepresented

---

## 🔮 Future Improvements

* Switch to RAF-DB (RGB dataset)
* Add attention mechanism (SE-Net)
* Temporal modeling (LSTM)
* On-device personalization

---

## 👤 Author

Made with ❤️ by **Riri Putri**

---
