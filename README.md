# Face and Eye Detection

## Objective

The purpose of this project:
1) Develop computer vision algorithms that can detect eyes, eye blinks and drowsiness. 
2) Implement the CV algorithms in a raspberry pi microprocessor with camera (Edge device) and capture the results on a video. 
3) Create a desktop GUI to detect eye blinks and drowsiness.

## Hardware

The hardware used for this project comprises:
1) Raspberry Pi 5 microprocessor, 4GB RAM
2) Raspberry Pi Camera Module V2 (RPI-CAM-V2) 

## Program

The programs were developed in Python. There are three python files: 
1) eye_detection.py - detects the eyes (implememted on both webcam and raspberry pi cam)
2) eyeblink_detection_2.py - detects and counts the number of eye blinks (implemented on both webcam and raspberry pi cam)
3) drowsiness_detect_2.py - contains functions that detects drowsiness and sounds alarm 
4) drowsiness_app.py - Tkinter (desktop) app that counts # of eye blinks and detects drowsiness

## Techniques

   - Data processing
   - Facial Features (Eyes, nose, mouth) detection 
   - Desktop app development 

## Algorithms 

   - Face mesh generation
   - Eye landmark detection and drawing
   - Calculate eye aspect ratio (EAR) - eye blinks and drowsiness detection  

## Libraries

   - OpenCV
   - picamera2
   - imutils
   - Mediapipe
   - NumPy
   - SciPy
   - time
   - playsound
   - threading
   - Tkinter (desktop app)
