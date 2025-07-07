//
//  CameraViewController.swift
//  SlipAngleSensor
//
//  Created by Abe Troop on 6/12/25.
//

import UIKit
import AVFoundation
import Photos

class CameraViewController: UIViewController, AVCaptureFileOutputRecordingDelegate {
    var captureSession: AVCaptureSession!
    var previewLayer: AVCaptureVideoPreviewLayer!
    var movieOutput: AVCaptureMovieFileOutput!
    let sensorLogger = SensorLogger()

    
    func setupCamera() {
        captureSession = AVCaptureSession()

        // Get the default camera
        guard let camera = AVCaptureDevice.default(for: .video) else {
            print("No camera available")
            return
        }

        // Turn the camera into an input
        guard let input = try? AVCaptureDeviceInput(device: camera),
              captureSession.canAddInput(input) else {
            print("Cannot create camera input")
            return
        }

        captureSession.addInput(input)
        
        do {
            try camera.lockForConfiguration()
            
            if let desiredFormat = camera.formats.first(where: { format in
                let desc = format.formatDescription
                let dimensions = CMVideoFormatDescriptionGetDimensions(desc)
                return dimensions.width == 1920 && dimensions.height == 1080 &&
                    format.videoSupportedFrameRateRanges.contains(where: { $0.maxFrameRate >= 240 })
            }) {
                camera.activeFormat = desiredFormat

                // Now set the actual frame rate you want
                camera.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 240)
                camera.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 240)
                print("Using 1080p @ 240 FPS")
            } else if let fallbackFormat = camera.formats.first(where: { format in
                let desc = format.formatDescription
                let dimensions = CMVideoFormatDescriptionGetDimensions(desc)
                return dimensions.width == 1280 && dimensions.height == 720 &&
                    format.videoSupportedFrameRateRanges.contains(where: { $0.maxFrameRate >= 240 })
            }) {
                camera.activeFormat = fallbackFormat
                camera.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 240)
                camera.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 240)
                print("Using 720p @ 240 FPS fallback")

            } else if let legacyFormat = camera.formats.first(where: { format in
                let desc = format.formatDescription
                let dimensions = CMVideoFormatDescriptionGetDimensions(desc)
                return dimensions.width == 1280 && dimensions.height == 720 &&
                    format.videoSupportedFrameRateRanges.contains(where: { $0.maxFrameRate >= 120 })
            }) {
                camera.activeFormat = legacyFormat
                camera.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 120)
                camera.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 120)
                print("Using 720p @ 120 FPS fallback")
            } else {
                print("No format found for 240 FPS — using default format and frame rate")
            }

            camera.unlockForConfiguration()
        } catch {
            print("Could not lock configuration: \(error)")
        }
        
        // Add movie file output
        movieOutput = AVCaptureMovieFileOutput()

        if captureSession.canAddOutput(movieOutput) {
            captureSession.addOutput(movieOutput)
        }

        // Create a preview layer to show the camera feed
        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.frame = view.bounds // Changes the size of the video (currently entire screen)
        previewLayer.videoGravity = .resizeAspectFill // Changes how the video fits on screen
        view.layer.insertSublayer(previewLayer, at: 0) // Adds the video preview behind everything

        // Start the camera
        DispatchQueue.global(qos: .userInitiated).async {
            self.captureSession.startRunning()
        }
    }
    
    func startRecording() {
        // Get a path to save the video (temporary location)
        let outputPath = NSTemporaryDirectory() + "output.mov"
        let outputFileURL = URL(fileURLWithPath: outputPath)

        // If there's already a video, remove it
        try? FileManager.default.removeItem(at: outputFileURL)

        // Start recording
        movieOutput.startRecording(to: outputFileURL, recordingDelegate: self)
        print("Started recording to: \(outputFileURL)")
        
        sensorLogger.startLogging()
    }
    
    func stopRecording() {
        if movieOutput.isRecording {
            movieOutput.stopRecording()
            print("Stopped recording.")
            
            sensorLogger.stopLogging()
            sensorLogger.saveToFile()
        }
    }
    
    func fileOutput(_ output: AVCaptureFileOutput,
                    didFinishRecordingTo outputFileURL: URL,
                    from connections: [AVCaptureConnection],
                    error: Error?) {
        if let error = error {
            print("Recording failed: \(error.localizedDescription)")
            return
        }

        // Save to photo library
        PHPhotoLibrary.shared().performChanges({
            PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: outputFileURL)
        }) { success, error in
            if success {
                print("Saved to photo library!")
            } else {
                print("Error saving to photo library: \(error?.localizedDescription ?? "unknown error")")
            }
        }
    }
    
    @objc func startRecordingFromSwiftUI() {
        startRecording()
    }

    @objc func stopRecordingFromSwiftUI() {
        stopRecording()
    }

    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        setupCamera()
        NotificationCenter.default.addObserver(self, selector: #selector(startRecordingFromSwiftUI), name: .startRecording, object: nil)
        NotificationCenter.default.addObserver(self, selector: #selector(stopRecordingFromSwiftUI), name: .stopRecording, object: nil)

    }

}

extension Notification.Name {
    static let startRecording = Notification.Name("startRecording")
    static let stopRecording = Notification.Name("stopRecording")
}

