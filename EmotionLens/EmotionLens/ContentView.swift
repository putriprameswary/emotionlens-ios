// ContentView.swift
// Root view: camera preview + face/emotion overlays + bottom panel.

import SwiftUI
import Combine

// MARK: - Detection State

enum DetectionState {
    case idle
    case noFace
    case detecting([FaceEmotionPair])
}

struct FaceEmotionPair: Identifiable {
    let id:     Int
    let face:   DetectedFace
    let result: EmotionResult?
}

// MARK: - ContentView

struct ContentView: View {
    @StateObject private var camera = CameraManager()
    @State private var state: DetectionState = .idle

    private let detector   = FaceDetector()
    private let classifier = EmotionClassifierWrapper()

    var body: some View {
        ZStack {
            // ── Background ──────────────────────────────────────────────
            Color.black.ignoresSafeArea()

            // ── Live camera preview ─────────────────────────────────────
            CameraPreviewView(cameraManager: camera)
                .ignoresSafeArea()

            // ── Per-face overlays ───────────────────────────────────────
            GeometryReader { geo in
                ForEach(activePairs, id: \.id) { pair in
                    EmotionOverlayView(pair: pair, screenSize: geo.size)
                }
            }

            // ── No-face banner ──────────────────────────────────────────
            if case .noFace = state {
                VStack {
                    Label("No face detected", systemImage: "person.slash")
                        .font(.system(size: 14, weight: .medium))
                        .foregroundColor(.white.opacity(0.85))
                        .padding(.horizontal, 14)
                        .padding(.vertical, 8)
                        .background(.ultraThinMaterial, in: Capsule())
                    Spacer()
                }
                .padding(.top, 60)
                .animation(.easeInOut(duration: 0.3), value: UUID())
            }

            // ── Bottom emotion panel ────────────────────────────────────
            VStack {
                Spacer()
                BottomPanelView(state: state)
            }
        }
        .onReceive(camera.$currentBuffer.compactMap { $0 }) { buffer in
            processFrame(buffer)
        }
    }

    // MARK: - Helpers

    private var activePairs: [FaceEmotionPair] {
        if case .detecting(let pairs) = state { return pairs }
        return []
    }

    /// Run face detection + emotion classification on each new camera frame.
    private func processFrame(_ buffer: CVPixelBuffer) {
        let size = camera.imageSize
        detector.detect(in: buffer, imageSize: size) { faces in
            DispatchQueue.main.async {
                guard !faces.isEmpty else {
                    self.state = .noFace
                    return
                }
                let pairs = faces.map { face -> FaceEmotionPair in
                    let result = self.classifier.classify(
                        pixelBuffer: buffer,
                        face: face,
                        imageSize: size
                    )
                    return FaceEmotionPair(id: face.index, face: face, result: result)
                }
                self.state = .detecting(pairs)
            }
        }
    }
}
