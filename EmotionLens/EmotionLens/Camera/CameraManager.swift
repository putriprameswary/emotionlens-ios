// CameraManager.swift
// Manages AVCaptureSession for front-camera video output.
// Delivers BGRA CVPixelBuffer at 1/3 frame rate to reduce CPU load.

import AVFoundation
import Combine

class CameraManager: NSObject, ObservableObject {
    @Published var currentBuffer: CVPixelBuffer?
    @Published var imageSize: CGSize = CGSize(width: 1280, height: 720)

    private(set) var previewLayer: AVCaptureVideoPreviewLayer?
    private var captureSession: AVCaptureSession?
    private let videoQueue = DispatchQueue(
        label: "com.emotionlens.video",
        qos: .userInitiated
    )

    // Frame throttle: only process every Nth frame
    private var frameCount   = 0
    private let frameInterval = 3   // ~10 fps from 30 fps camera

    // MARK: - Setup

    func setup() {
        let session = AVCaptureSession()
        session.beginConfiguration()
        session.sessionPreset = .hd1280x720

        // ── Front camera input ──────────────────────────────────────────
        guard let device = AVCaptureDevice.default(
            .builtInWideAngleCamera,
            for: .video,
            position: .front
        ) else {
            print("CameraManager: front camera not available")
            session.commitConfiguration()
            return
        }

        guard let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input) else {
            print("CameraManager: cannot create device input")
            session.commitConfiguration()
            return
        }
        session.addInput(input)

        // ── Video data output ──────────────────────────────────────────
        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: videoQueue)

        guard session.canAddOutput(output) else {
            print("CameraManager: cannot add video output")
            session.commitConfiguration()
            return
        }
        session.addOutput(output)

        // ── Connection orientation ─────────────────────────────────────
        if let connection = output.connection(with: .video) {
            if connection.isVideoOrientationSupported {
                connection.videoOrientation = .portrait
            }
            // Do NOT mirror here — FaceDetector handles mirroring in Vision coords
            connection.isVideoMirrored = false
        }

        session.commitConfiguration()

        self.previewLayer = AVCaptureVideoPreviewLayer(session: session)
        self.previewLayer?.videoGravity = .resizeAspectFill
        self.captureSession = session
    }

    // MARK: - Lifecycle

    func start() {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.captureSession?.startRunning()
        }
    }

    func stop() {
        captureSession?.stopRunning()
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {

    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        frameCount += 1
        guard frameCount % frameInterval == 0 else { return }
        guard let buffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        // Publish on main thread so SwiftUI can observe
        DispatchQueue.main.async { [weak self] in
            self?.currentBuffer = buffer
        }
    }

    func captureOutput(
        _ output: AVCaptureOutput,
        didDrop sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        // Silently drop — alwaysDiscardsLateVideoFrames handles backpressure
    }
}
