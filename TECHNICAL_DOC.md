# Technical Documentation

# EmotionLens: Real-Time Facial Emotion Recognition on iOS Using Core ML and MobileNetV2
---

# 1. Introduction

EmotionLens is an end-to-end facial emotion recognition system designed to perform real-time emotion classification directly on iOS devices without requiring internet connectivity.

The project was developed to explore the complete machine learning deployment pipeline, starting from dataset preparation and model training to mobile deployment using Apple's Core ML ecosystem.

Unlike many experimental machine learning projects that remain inside Jupyter notebooks, EmotionLens focuses on practical deployment. The trained model is converted into Core ML format and integrated into a SwiftUI application capable of detecting facial emotions from a live camera feed.

The project uses a lightweight convolutional neural network architecture based on the CLCM (Custom Lightweight CNN-based Model) framework proposed by Gursesli et al. (2024), which utilizes MobileNetV2 as the feature extraction backbone.

The final application supports five emotion classes:

* Angry
* Happy
* Neutral
* Sad
* Surprise

All inference is executed locally on the device using Core ML, ensuring low latency, privacy preservation, and offline functionality.

---

# 2. Project Objectives

The main objectives of EmotionLens are:

1. Develop a lightweight facial emotion recognition model suitable for mobile devices.
2. Implement transfer learning using MobileNetV2 for emotion classification.
3. Deploy the trained model to the Apple ecosystem through Core ML.
4. Build a real-time emotion recognition application using SwiftUI.
5. Evaluate the strengths and limitations of FER-2013 for practical deployment scenarios.
6. Analyze the gap between validation performance and real-world application behavior.

---

# 3. Background and Motivation

Facial Emotion Recognition (FER) is a computer vision task that aims to classify human facial expressions into predefined emotional categories.

FER has numerous potential applications:

* Human-computer interaction
* Mental health monitoring
* Educational technology
* Accessibility systems
* Adaptive user interfaces
* Social robotics

Although FER has been extensively studied, deploying FER models on mobile devices remains challenging because of computational limitations and the requirement for real-time inference.

The Apple ecosystem provides several technologies that make on-device FER practical:

* Core ML for machine learning inference
* Vision Framework for face detection
* AVFoundation for camera access
* Metal acceleration for hardware optimization
* SwiftUI for modern user interfaces

EmotionLens was developed as an exploration of how these technologies can be combined into a complete machine learning product.

---

# 4. Reference Paper

The primary reference used in this project is:

Gursesli, M. C., Lombardi, S., Duradoni, M., Bocchi, L., Guazzini, A., & Lanata, A. (2024).

Facial Emotion Recognition (FER) Through Custom Lightweight CNN Model: Performance Evaluation in Public Datasets.

IEEE Access, Volume 12.

DOI:
10.1109/ACCESS.2024.3380847

The paper was selected because:

* It focuses on lightweight architectures.
* It is designed for edge and mobile deployment.
* It uses MobileNetV2 as its backbone.
* It evaluates performance on FER-2013.
* It provides reproducible experimental results.

The paper reported approximately 63% accuracy on FER-2013 using seven emotion classes.

EmotionLens achieved 64.8% validation accuracy using a five-class subset of the same dataset.

---

# 5. Dataset

## 5.1 FER-2013

EmotionLens uses the FER-2013 dataset.

FER-2013 is one of the most widely used benchmark datasets for facial emotion recognition research.

Dataset characteristics:

| Property         | Value     |
| ---------------- | --------- |
| Dataset          | FER-2013  |
| Image Type       | Grayscale |
| Resolution       | 48×48     |
| Total Images     | ~30,000   |
| Original Classes | 7         |
| Used Classes     | 5         |

Original FER-2013 classes:

* Angry
* Disgust
* Fear
* Happy
* Neutral
* Sad
* Surprise

---

## 5.2 Class Selection

Only five classes were used:

* Angry
* Happy
* Neutral
* Sad
* Surprise

The following classes were removed:

* Disgust
* Fear

During earlier experiments, disgust was merged into angry and fear was merged into surprise.

This approach introduced label contamination because the merged classes still contained visual characteristics that differed from their new labels.

The resulting model plateaued at approximately 58% validation accuracy.

To address this issue, disgust and fear samples were removed entirely.

This decision improved class purity and increased validation accuracy to 64.8%.

---

## 5.3 Dataset Distribution

| Class    | Training | Validation |
| -------- | -------- | ---------- |
| Angry    | 3995     | 958        |
| Happy    | 7215     | 1774       |
| Neutral  | 4965     | 1233       |
| Sad      | 4830     | 1247       |
| Surprise | 3171     | 831        |

Total samples:

* Training: 24,176
* Validation: 6,043

The dataset is moderately imbalanced, with happy containing more than twice as many samples as surprise.

This imbalance influences model behavior during training and evaluation.

---

# 6. Data Preprocessing

Several preprocessing steps were applied before training.

## 6.1 Grayscale to RGB Conversion

FER-2013 images are stored in grayscale format.

MobileNetV2 expects RGB input.

To satisfy this requirement, grayscale images were converted into RGB by duplicating the grayscale channel three times.

This technique is commonly used when transferring grayscale datasets into RGB-based pretrained architectures.

Example:

Gray Channel → [G]

Converted RGB:

R = G
G = G
B = G

Although no new color information is introduced, this allows compatibility with ImageNet-pretrained weights.

---

## 6.2 Image Resizing

Original image size:

48 × 48

Training image size:

96 × 96

Bilinear interpolation was used during resizing.

The larger resolution provides more spatial information for MobileNetV2 while maintaining reasonable computational requirements.

---

## 6.3 Normalization

Pixel values were normalized to:

0.0 – 1.0

using:

pixel = pixel / 255.0

Normalization improves optimization stability and convergence speed.

---

## 6.4 Data Augmentation

To improve generalization, the following augmentations were applied:

* Horizontal flip
* Rotation (±15°)
* Zoom (±10%)
* Brightness adjustment (±20%)

The objective was to simulate real-world variations encountered during mobile deployment.

---

## 6.5 Class Weight Balancing

Class weights were computed to reduce the impact of dataset imbalance.

Without balancing, the model tends to over-predict majority classes such as happy and neutral.

Class weighting encourages the model to pay greater attention to minority classes such as surprise and angry.

---

# 7. Model Architecture

## 7.1 Overview

EmotionLens uses a modified CLCM architecture based on MobileNetV2.

The architecture consists of:

1. MobileNetV2 feature extractor
2. Global average pooling layer
3. Dense classification head
4. Softmax output layer

The design prioritizes:

* Lightweight computation
* Mobile deployment
* Fast inference
* Small model size

---

## 7.2 Backbone Network

Backbone:

MobileNetV2

Configuration:

* Alpha = 0.75
* ImageNet pretrained
* include_top = False

MobileNetV2 was selected because:

* Low parameter count
* Efficient depthwise separable convolutions
* Strong mobile deployment performance
* Native compatibility with Core ML

---

## 7.3 Classification Head

The classifier consists of:

GlobalAveragePooling2D

↓

Dense(256, ReLU)

↓

BatchNormalization

↓

Dropout(0.4)

↓

Dense(128, ReLU)

↓

Dropout(0.3)

↓

Dense(5, Softmax)

The dropout layers reduce overfitting while maintaining model capacity.

---

## 7.4 Model Statistics

| Property       | Value              |
| -------------- | ------------------ |
| Parameters     | 1.74 Million       |
| Input Size     | 96×96×3            |
| Output Classes | 5                  |
| Model Size     | ~3.6 MB            |
| Backbone       | MobileNetV2 α=0.75 |

The final model size is sufficiently small for real-time mobile deployment.
