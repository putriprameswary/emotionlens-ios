// EmotionOverlayView.swift
// Draws a bounding box + emoji + confidence label above each detected face.

import SwiftUI

struct EmotionOverlayView: View {
    let pair:       FaceEmotionPair
    let screenSize: CGSize

    // Convert normalised bounding box to screen-space CGRect
    private var rect: CGRect {
        CGRect(
            x:      pair.face.boundingBox.origin.x * screenSize.width,
            y:      pair.face.boundingBox.origin.y * screenSize.height,
            width:  pair.face.boundingBox.width    * screenSize.width,
            height: pair.face.boundingBox.height   * screenSize.height
        )
    }

    private var boxColor: Color {
        pair.result.map { $0.emotion.color } ?? .white.opacity(0.5)
    }

    var body: some View {
        ZStack {
            // ── Bounding box ────────────────────────────────────────────
            RoundedRectangle(cornerRadius: 8)
                .stroke(boxColor, lineWidth: 2.5)
                .shadow(color: boxColor.opacity(0.6), radius: 4)
                .frame(width: rect.width, height: rect.height)
                .position(x: rect.midX, y: rect.midY)

            // ── Emoji + confidence badge above the box ──────────────────
            if let result = pair.result {
                VStack(spacing: 2) {
                    Text(result.emotion.emoji)
                        .font(.system(size: 40))
                        .shadow(radius: 2)

                    Text("\(Int(result.confidence * 100))%")
                        .font(.system(size: 12, weight: .bold, design: .monospaced))
                        .foregroundColor(result.emotion.color)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(.black.opacity(0.45), in: Capsule())
                }
                .position(x: rect.midX, y: rect.minY - 44)
                .transition(.opacity.combined(with: .scale(scale: 0.85)))
                .animation(.easeInOut(duration: 0.2), value: result.emotion)
            }
        }
    }
}
