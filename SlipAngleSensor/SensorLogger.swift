//
//  SensorLogger.swift
//  SlipAngleSensor
//
//  Created by Abe Troop on 6/15/25.
//


import Foundation
import CoreMotion
import CoreLocation

class SensorLogger: NSObject, CLLocationManagerDelegate {
    private let motionManager = CMMotionManager()
    private let locationManager = CLLocationManager()

    private(set) var imuData: [[String: Any]] = []
    private var currentLocation: CLLocation?
    
    private var startTime: TimeInterval?

    override init() {
        super.init()
        locationManager.delegate = self
    }

    func startLogging() {
        imuData = []
        locationManager.requestWhenInUseAuthorization()
        locationManager.startUpdatingLocation()
        
        startTime = Date().timeIntervalSince1970

        if motionManager.isDeviceMotionAvailable {
            motionManager.deviceMotionUpdateInterval = 1.0 / 60.0
            motionManager.startDeviceMotionUpdates(to: .main) { [weak self] motion, _ in
                guard let self = self, let motion = motion else { return }
                let timestamp = Date().timeIntervalSince1970
                let entry: [String: Any] = [
                    "timestamp": timestamp,
                    "acceleration": [
                        "x": motion.userAcceleration.x,
                        "y": motion.userAcceleration.y,
                        "z": motion.userAcceleration.z
                    ],
                    "rotationRate": [
                        "x": motion.rotationRate.x,
                        "y": motion.rotationRate.y,
                        "z": motion.rotationRate.z
                    ],
                    "attitude": [
                        "roll": motion.attitude.roll,
                        "pitch": motion.attitude.pitch,
                        "yaw": motion.attitude.yaw
                    ],
                    "location": [
                        "lat": self.currentLocation?.coordinate.latitude ?? 0.0,
                        "lon": self.currentLocation?.coordinate.longitude ?? 0.0,
                        "speed": self.currentLocation?.speed ?? 0.0,
                        "alt": self.currentLocation?.altitude ?? 0.0
                    ]
                ]
                self.imuData.append(entry)
            }
        }
    }

    func stopLogging() {
        motionManager.stopDeviceMotionUpdates()
        locationManager.stopUpdatingLocation()
    }

    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        currentLocation = locations.last
    }

    func saveToFile() {
        let filename = getDocumentsDirectory().appendingPathComponent("imu_data.json")
        do {
            let payload: [String: Any] = [
                "video_start_timestamp": startTime ?? Date().timeIntervalSince1970,
                "samples": imuData
            ]
            let data = try JSONSerialization.data(withJSONObject: payload, options: .prettyPrinted)
            try data.write(to: filename)
            print("Sensor data saved to: \(filename)")
        } catch {
            print("Failed to save sensor data: \(error)")
        }
    }

    private func getDocumentsDirectory() -> URL {
        return FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }
}
