//
//  CameraView.swift
//  SlipAngleSensor
//
//  Created by Abe Troop on 6/14/25.
//

import SwiftUI

struct CameraView: View {
    @State private var isRecording = false
    
    var body: some View {
        ZStack {
            CameraViewControllerRepresentable()
                .edgesIgnoringSafeArea(.all)
                .onReceive(NotificationCenter.default.publisher(for: .stopRecording)) { _ in
                        isRecording = false
                }

            VStack {
                Spacer()

                if isRecording {
                    Text("Recording...")
                        .foregroundColor(.red)
                        .font(.headline)
                        .padding(.bottom, 10)

                    Button(action: {
                        NotificationCenter.default.post(name: .stopRecording, object: nil)
                        isRecording = false
                    }) {
                        Text("Stop")
                            .padding()
                            .frame(maxWidth: 120)
                            .background(Color.red)
                            .foregroundColor(.white)
                            .cornerRadius(12)
                    }
                } else {
                    Button(action: {
                        NotificationCenter.default.post(name: .startRecording, object: nil)
                        isRecording = true
                    }) {
                        Text("Start")
                            .padding()
                            .frame(maxWidth: 120)
                            .background(Color.green)
                            .foregroundColor(.white)
                            .cornerRadius(12)
                    }
                }

                Spacer().frame(height: 40)
            }
            .padding(.horizontal)
        }
    }
}

struct CameraViewControllerRepresentable: UIViewControllerRepresentable {
    func makeUIViewController(context: Context) -> CameraViewController {
        return CameraViewController()
    }

    func updateUIViewController(_ uiViewController: CameraViewController, context: Context) {}
}
