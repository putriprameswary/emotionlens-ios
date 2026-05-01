// ConfidenceBarView.swift
// Horizontal animated confidence bar for a single emotion class.

import SwiftUI

struct ConfidenceBarView: View {
    let emotion:    Emotion
    let confidence: Float

    var body: some View {
        HStack(spacing: 8) {
            // Emoji icon (fixed width)
            Text(emotion.emoji)
                .font(.system(size: 15))
                .frame(width: 22, alignment: .center)

            // Class name (fixed width, left-aligned)
            Text(emotion.displayName)
                .font(.system(size: 12, weight: .medium))
                .foregroundColor(.primary)
                .frame(width: 64, alignment: .leading)
                .lineLimit(1)

            // Progress bar (fills remaining space)
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    // Track
                    RoundedRectangle(cornerRadius: 3)
                        .fill(Color.white.opacity(0.12))
                        .frame(height: 8)

                    // Fill
                    RoundedRectangle(cornerRadius: 3)
                        .fill(
                            LinearGradient(
                                colors: [emotion.color.opacity(0.75), emotion.color],
                                startPoint: .leading,
                                endPoint: .trailing
                            )
                        )
                        .frame(
                            width: max(0, CGFloat(confidence) * geo.size.width),
                            height: 8
                        )
                        .animation(.easeInOut(duration: 0.2), value: confidence)
                }
            }
            .frame(height: 8)

            // Percentage label (fixed width, right-aligned)
            Text("\(Int(confidence * 100))%")
                .font(.system(size: 11, weight: .medium, design: .monospaced))
                .foregroundColor(.secondary)
                .frame(width: 34, alignment: .trailing)
        }
    }
}
