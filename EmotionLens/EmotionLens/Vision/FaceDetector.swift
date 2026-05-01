// FaceDetector.swift
// Detects faces in a CVPixelBuffer using Vision framework.
//
// Coordinate system conversion:
//   Vision uses bottom-left origin + no front-camera mirroring.
//   We convert to UIKit top-left origin + mirror X for front camera.

import Vision
import UIKit

class FaceDetector {

    /// Detect faces in the given pixel buffer and return normalized DetectedFace objects.
    /// - Parameters:
    ///   - pixelBuffer: The camera frame (BGRA, front camera).
    ///   - imageSize: The pixel dimensions of the frame (e.g. 1280×720).
    ///   - completion: Called on an arbitrary queue with the detected faces.
    func detect(
        in pixelBuffer: CVPixelBuffer,
        imageSize: CGSize,
        completion: @escaping ([DetectedFace]) -> Void
    ) {
        let request = VNDetectFaceRectanglesRequest { req, error in
            if let error = error {
                print("FaceDetector error: \(error)")
                completion([])
                return
            }
            guard let observations = req.results as? [VNFaceObservation],
                  !observations.isEmpty else {
                completion([])
                return
            }
            let faces = observations.enumerated().compactMap { (idx, obs) -> DetectedFace? in
                self.convertBbox(obs.boundingBox, index: idx, imageSize: imageSize)
            }
            completion(faces)
        }

        // .leftMirrored accounts for the front camera being mirrored in AVFoundation
        let handler = VNImageRequestHandler(
            cvPixelBuffer: pixelBuffer,
            orientation: .leftMirrored,
            options: [:]
        )
        do {
            try handler.perform([request])
        } catch {
            print("FaceDetector VNImageRequestHandler error: \(error)")
            completion([])
        }
    }

    // MARK: - Coordinate Conversion

    /// Convert a Vision bounding box (bottom-left, 0–1) to UIKit coords (top-left, 0–1)
    /// with front-camera X mirroring, 15% padding, square crop, and clamping.
    private func convertBbox(
        _ bbox: CGRect,
        index: Int,
        imageSize: CGSize
    ) -> DetectedFace? {
        // 1. Flip Y: Vision origin is bottom-left, UIKit is top-left
        let flippedY  = 1.0 - bbox.origin.y - bbox.height
        // 2. Mirror X: front camera is mirrored so flip horizontal axis
        let mirroredX = 1.0 - bbox.origin.x - bbox.width

        // 3. Add 15% padding on all sides
        let padX = bbox.width  * 0.15
        let padY = bbox.height * 0.15
        var padded = CGRect(
            x:      mirroredX - padX,
            y:      flippedY  - padY,
            width:  bbox.width  + 2 * padX,
            height: bbox.height + 2 * padY
        )

        // 4. Force square crop (use the larger dimension)
        let side = max(padded.width, padded.height)
        padded = CGRect(
            x:      padded.midX - side / 2,
            y:      padded.midY - side / 2,
            width:  side,
            height: side
        )

        // 5. Clamp to [0, 1] unit rectangle
        let unitRect  = CGRect(x: 0, y: 0, width: 1, height: 1)
        let clamped   = padded.intersection(unitRect)
        guard !clamped.isNull, !clamped.isInfinite,
              clamped.width > 0.02, clamped.height > 0.02 else { return nil }

        // 6. Convert normalised coords to pixel rect for cropping
        let pixelRect = CGRect(
            x:      clamped.origin.x * imageSize.width,
            y:      clamped.origin.y * imageSize.height,
            width:  clamped.width    * imageSize.width,
            height: clamped.height   * imageSize.height
        )

        return DetectedFace(
            boundingBox: clamped,
            pixelRect:   pixelRect,
            index:       index
        )
    }
}
