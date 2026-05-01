// EmotionModel.swift
// Shared data types for the EmotionLens pipeline.
//
// Architecture reference:
// Gursesli et al. (2024). "Facial Emotion Recognition (FER) Through Custom
// Lightweight CNN Model." IEEE Access, 12, 45543–45559.
// DOI: 10.1109/ACCESS.2024.3380847

import SwiftUI

// MARK: - Emotion Enum

enum Emotion: String, CaseIterable, Hashable {
    // Order MUST match Python CLASS_LABELS = ['angry','happy','neutral','sad','surprise']
    case angry    = "angry"
    case happy    = "happy"
    case neutral  = "neutral"
    case sad      = "sad"
    case surprise = "surprise"

    var emoji: String {
        switch self {
        case .angry:    return "😠"
        case .happy:    return "😊"
        case .neutral:  return "😐"
        case .sad:      return "😢"
        case .surprise: return "😲"
        }
    }

    var color: Color {
        switch self {
        case .angry:    return Color(red: 0.957, green: 0.263, blue: 0.212)  // Material Red
        case .happy:    return Color(red: 0.298, green: 0.686, blue: 0.314)  // Material Green
        case .neutral:  return Color(red: 0.620, green: 0.620, blue: 0.620)  // Gray
        case .sad:      return Color(red: 0.129, green: 0.588, blue: 0.953)  // Material Blue
        case .surprise: return Color(red: 1.000, green: 0.596, blue: 0.000)  // Amber
        }
    }

    var displayName: String { rawValue.capitalized }

    /// Case-insensitive lookup from the classLabel string returned by CoreML.
    static func from(_ string: String) -> Emotion? {
        Emotion.allCases.first { $0.rawValue == string.lowercased() }
    }
}

// MARK: - Result Types

struct EmotionResult {
    let emotion:    Emotion
    let confidence: Float
    let allScores:  [Emotion: Float]  // smoothed probabilities for all 5 classes
    let faceIndex:  Int
}

struct DetectedFace {
    let boundingBox: CGRect  // normalized, UIKit coords (origin top-left, 0–1)
    let pixelRect:   CGRect  // pixel coords in image space
    let index:       Int
}
