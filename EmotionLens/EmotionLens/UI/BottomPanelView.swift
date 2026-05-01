// BottomPanelView.swift
// Shows per-emotion confidence bars (single face) or cards (multi-face).

import SwiftUI

// MARK: - Router

struct BottomPanelView: View {
    let state: DetectionState

    var body: some View {
        switch state {
        case .idle, .noFace:
            EmptyView()

        case .detecting(let pairs):
            let active = pairs.filter { $0.result != nil }
            if active.isEmpty {
                EmptyView()
            } else if active.count == 1, let r = active[0].result {
                SingleFacePanel(result: r)
            } else {
                MultiFacePanel(pairs: active)
            }
        }
    }
}

// MARK: - Single Face Panel

struct SingleFacePanel: View {
    let result: EmotionResult

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            // Header
            HStack(spacing: 10) {
                Text(result.emotion.emoji)
                    .font(.system(size: 36))
                Text(result.emotion.displayName)
                    .font(.system(size: 26, weight: .bold, design: .rounded))
                    .foregroundColor(result.emotion.color)
                Spacer()
                Text("\(Int(result.confidence * 100))%")
                    .font(.system(size: 18, weight: .semibold, design: .monospaced))
                    .foregroundColor(result.emotion.color.opacity(0.9))
            }

            Divider()
                .background(Color.white.opacity(0.2))

            // Per-emotion bars
            ForEach(Emotion.allCases, id: \.self) { emotion in
                ConfidenceBarView(
                    emotion:    emotion,
                    confidence: result.allScores[emotion] ?? 0
                )
            }
        }
        .padding(18)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 22))
        .padding(.horizontal, 16)
        .padding(.bottom, 30)
        .animation(.easeInOut(duration: 0.25), value: result.emotion)
    }
}

// MARK: - Multi-Face Panel

struct MultiFacePanel: View {
    let pairs: [FaceEmotionPair]

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 10) {
                ForEach(pairs) { pair in
                    if let r = pair.result {
                        FaceCard(faceIndex: pair.face.index, result: r)
                    }
                }
            }
            .padding(.horizontal, 16)
        }
        .padding(.bottom, 30)
    }
}

struct FaceCard: View {
    let faceIndex: Int
    let result:    EmotionResult

    var body: some View {
        VStack(spacing: 6) {
            Text("Face \(faceIndex + 1)")
                .font(.caption2)
                .foregroundColor(.secondary)
            Text(result.emotion.emoji)
                .font(.system(size: 34))
            Text(result.emotion.displayName)
                .font(.system(size: 13, weight: .bold))
                .foregroundColor(result.emotion.color)
            Text("\(Int(result.confidence * 100))%")
                .font(.system(size: 11, design: .monospaced))
                .foregroundColor(.secondary)
        }
        .frame(width: 88)
        .padding(12)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
    }
}
