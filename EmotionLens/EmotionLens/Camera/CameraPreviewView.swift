// CameraPreviewView.swift
// UIViewRepresentable bridge that embeds AVCaptureVideoPreviewLayer in SwiftUI.

import SwiftUI
import AVFoundation

struct CameraPreviewView: UIViewRepresentable {
    let cameraManager: CameraManager

    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: .zero)
        view.backgroundColor = .black

        // Set up the session and attach preview layer before returning
        cameraManager.setup()

        if let layer = cameraManager.previewLayer {
            layer.frame = view.bounds
            layer.videoGravity = .resizeAspectFill
            view.layer.addSublayer(layer)
        }

        cameraManager.start()
        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {
        // Keep preview layer filling the view as geometry changes (rotation, resize)
        DispatchQueue.main.async {
            cameraManager.previewLayer?.frame = uiView.bounds
        }
    }
}
