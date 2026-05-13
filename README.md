## Overview

This project is an AI-based driver fatigue detection system. The goal is to detect signs of driver fatigue using a smartphone camera and send a Telegram notification when fatigue is confirmed.

The system uses a hybrid approach that combines:

- CNN image classification
- MediaPipe facial landmarks
- Eye Aspect Ratio (EAR)
- Mouth Aspect Ratio (MAR)
- Head-pose indicators
- Calibration
- Temporal validation

The model was first trained and tested offline, then tested in real time using Python before being converted and integrated into the mobile application.

## Problem

Driver fatigue is a serious road-safety issue, especially during long-distance, night-time, or monotonous driving. Fatigue cannot be measured directly using a normal camera, so the system must infer it from indirect visual cues such as:

- Eye closure
- Yawning
- Head posture
- Reduced facial responsiveness

A single image is not reliable enough because blinking, speaking, or temporary occlusion can lead to false detections. For this reason, the system uses multiple cues and validates fatigue over time.

## Main Features

- Real-time fatigue detection using a camera
- CNN-based drowsiness classification
- Facial landmark analysis using MediaPipe
- EAR calculation for eye closure detection
- MAR calculation for yawning detection
- Head-pose analysis
- Personalized calibration phase
- Temporal validation to reduce false alarms
- TensorFlow Lite model export for mobile integration
- Telegram notification when fatigue is confirmed

## System Pipeline

The system follows this pipeline:

1. Capture video frame from the camera
2. Detect and extract the driver's face
3. Convert the image to grayscale
4. Resize the image to 128x128
5. Run CNN inference on sampled frames
6. Extract facial landmarks using MediaPipe
7. Compute EAR, MAR, and head-pose indicators
8. Fuse CNN output with landmark-based cues
9. Apply temporal validation
10. Confirm fatigue if the fatigue score remains high over time
11. Send Telegram notification when fatigue is detected

## Dataset

The dataset was built using two sources:

1. Public fatigue-related image datasets
2. Custom in-car videos recorded by team members

The public dataset provided initial DROWSY and NATURAL examples. However, many images were duplicate or near-duplicate frames extracted from videos. These repeated images were removed to reduce memorization and improve generalization.

The custom videos were recorded in a car environment to make the dataset closer to the final use case. Frames were extracted from the videos, labeled, and merged with the cleaned public dataset.

## Preprocessing

Before training, all images were processed as follows:

- Face region extraction
- Resize to 128x128 pixels
- Grayscale conversion
- Normalization to the range [0, 1]
- Small data augmentations

Grayscale was used because the main fatigue cues are mostly based on shape and intensity, such as eye closure, mouth opening, and facial structure. It also reduces computation, which is useful for real-time mobile deployment.

## Models Tested

Three models were trained and compared:

| Model | Accuracy | DROWSY Recall | Decision |
|---|---:|---:|---|
| Model 1, Simple CNN | 80.99% | 93% | Baseline |
| Model 2, Deeper CNN | 83.33% | 96% | Selected |
| Model 3, MobileNetV2 | 83.85% | 83% | Not selected |

Although MobileNetV2 had the highest overall accuracy, it was not selected because it missed more DROWSY cases. In this project, DROWSY recall is more important than global accuracy because missing a fatigued driver is the most dangerous error.

Model 2 was selected because it achieved the best DROWSY recall.

## Why CNN and Facial Landmarks

The CNN and facial landmarks work independently at first, then their outputs are combined in the decision logic.

The CNN gives a drowsiness probability based on the face image.

MediaPipe landmarks provide measurable fatigue cues:

- EAR for eye closure
- MAR for mouth opening
- Head-pose indicators for nodding or head-down behavior

The final decision is produced using late fusion. This means the CNN score and landmark-based cues are combined into one fatigue score, then checked over time before declaring fatigue.

## Python Real-Time Testing

Before converting the model to the mobile application, the system was tested in Python using a live camera stream.

This step was important because offline model accuracy alone was not enough. Real-time testing showed that single-frame predictions could be unstable due to:

- Blinking
- Speaking
- Yawning
- Hand movement near the face
- Temporary occlusion
- Lighting changes

To improve stability, the system was updated with:

- Sampled-frame CNN inference
- Smoothed CNN output
- Facial landmark cues
- Driver-specific calibration
- Temporal validation

## How to Test the Model

To test the model in real time, run:



```bash
python live_test_model.py
```

## Note About Running the Android App

You can also run the Android application code, but you need an Android phone to install and test the app properly. The app uses the phone camera for live fatigue detection and communicates with the backend for Telegram notification.

Please note that our hosted database/backend service may expire soon, so some backend features such as login, fatigue logs, or Telegram notification may stop working unless the backend and database are redeployed or reconfigured.
