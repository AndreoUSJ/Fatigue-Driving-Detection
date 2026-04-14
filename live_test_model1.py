import cv2
import numpy as np
import tensorflow as tf
import time
import math
from collections import deque

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =========================================
# SETTINGS
# =========================================
CNN_MODEL_PATH = "model_1_simple_cnn.keras"
FACE_LANDMARKER_MODEL_PATH = "face_landmarker.task"

IMG_HEIGHT = 128
IMG_WIDTH = 128

# IMPORTANT:
# Confirm your class order from training:
# print(train_gen.class_indices)
# Example: {'DROWSY': 0, 'NATURAL': 1}
CLASS_NAMES = ['DROWSY', 'NATURAL']

# -----------------------------------------
# CNN temporal settings
# -----------------------------------------
EMA_ALPHA = 0.30
CNN_DROWSY_THRESHOLD = 0.65

# -----------------------------------------
# Eye closure settings
# -----------------------------------------
# EAR threshold:
# lower EAR = more closed eye
# start around 0.18 to 0.23 and tune
EAR_CLOSED_THRESHOLD = 0.20

# Consecutive frames with low EAR before treating as true eye closure
EYE_CLOSED_CONSEC_FRAMES = 6

# -----------------------------------------
# Final fusion / fatigue logic
# -----------------------------------------
WINDOW_SIZE = 20
WINDOW_ALERT_RATIO = 0.65
FATIGUE_CONSEC_TRIGGER = 10
FRAME_SKIP = 2
ALERT_HOLD_SEC = 2.0

SHOW_DEBUG = True

# =========================================
# MEDIAPIPE SETUP
# =========================================
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
RunningMode = vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=FACE_LANDMARKER_MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
face_landmarker = FaceLandmarker.create_from_options(options)

# =========================================
# LOAD CNN
# =========================================
cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
print("CNN model loaded successfully.")
print("MediaPipe Face Landmarker loaded successfully.")

# =========================================
# HELPERS
# =========================================
def euclidean(p1, p2):
    return math.dist(p1, p2)

def landmark_to_pixel(landmark, width, height):
    return (landmark.x * width, landmark.y * height)

# Common Face Mesh-style eye landmark sets used for EAR-like measurement.
# These are practical landmark choices for eyelid opening estimation.
LEFT_EYE = {
    "left_corner": 33,
    "right_corner": 133,
    "top1": 159,
    "bottom1": 145,
    "top2": 158,
    "bottom2": 153,
}

RIGHT_EYE = {
    "left_corner": 362,
    "right_corner": 263,
    "top1": 386,
    "bottom1": 374,
    "top2": 385,
    "bottom2": 380,
}

def compute_eye_ear(face_landmarks, eye_def, width, height):
    pts = {}
    for name, idx in eye_def.items():
        lm = face_landmarks[idx]
        pts[name] = landmark_to_pixel(lm, width, height)

    horizontal = euclidean(pts["left_corner"], pts["right_corner"])
    vertical1 = euclidean(pts["top1"], pts["bottom1"])
    vertical2 = euclidean(pts["top2"], pts["bottom2"])

    if horizontal == 0:
        return None

    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

def get_eye_features_from_landmarks(face_landmarks, width, height):
    left_ear = compute_eye_ear(face_landmarks, LEFT_EYE, width, height)
    right_ear = compute_eye_ear(face_landmarks, RIGHT_EYE, width, height)

    if left_ear is None or right_ear is None:
        return None, None, None

    avg_ear = (left_ear + right_ear) / 2.0
    return left_ear, right_ear, avg_ear

def preprocess_for_cnn(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_WIDTH, IMG_HEIGHT))
    img_array = resized.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=-1)   # (H,W,1)
    img_array = np.expand_dims(img_array, axis=0)    # (1,H,W,1)
    return img_array

def crop_face_from_landmarks(frame, face_landmarks):
    h, w = frame.shape[:2]

    xs = [lm.x for lm in face_landmarks]
    ys = [lm.y for lm in face_landmarks]

    x_min = max(0, int(min(xs) * w) - 20)
    y_min = max(0, int(min(ys) * h) - 20)
    x_max = min(w, int(max(xs) * w) + 20)
    y_max = min(h, int(max(ys) * h) + 20)

    if x_max <= x_min or y_max <= y_min:
        return frame

    return frame[y_min:y_max, x_min:x_max]

# =========================================
# VIDEO
# =========================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam.")
    raise SystemExit

print("Press 'q' to quit.")

frame_index = 0

ema_drowsy_prob = None
consec_cnn_drowsy = 0
consec_eye_closed = 0
fusion_window = deque(maxlen=WINDOW_SIZE)

last_alert_time = 0.0
current_status = "NATURAL"
current_confidence = 0.0

last_avg_ear = 0.0
last_cnn_drowsy_prob = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from webcam.")
        break

    frame = cv2.flip(frame, 1)
    display = frame.copy()
    frame_index += 1

    h, w = frame.shape[:2]

    if frame_index % FRAME_SKIP == 0:
        # MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int(time.time() * 1000)
        result = face_landmarker.detect_for_video(mp_image, timestamp_ms)

        face_found = result.face_landmarks is not None and len(result.face_landmarks) > 0

        if face_found:
            face_landmarks = result.face_landmarks[0]

            # Draw landmarks lightly
            for lm in face_landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(display, (x, y), 1, (255, 255, 0), -1)

            # -------- Eye closure branch --------
            left_ear, right_ear, avg_ear = get_eye_features_from_landmarks(face_landmarks, w, h)
            if avg_ear is not None:
                last_avg_ear = avg_ear
                eye_closed_now = avg_ear < EAR_CLOSED_THRESHOLD
            else:
                eye_closed_now = False

            if eye_closed_now:
                consec_eye_closed += 1
            else:
                consec_eye_closed = max(0, consec_eye_closed - 1)

            # -------- CNN branch --------
            face_crop = crop_face_from_landmarks(frame, face_landmarks)
            cnn_input = preprocess_for_cnn(face_crop)

            pred = float(cnn_model.predict(cnn_input, verbose=0)[0][0])

            # Assuming sigmoid output for class index 1
            # With CLASS_NAMES = ['DROWSY', 'NATURAL']
            natural_prob = pred
            drowsy_prob = 1.0 - natural_prob
            last_cnn_drowsy_prob = drowsy_prob

            if ema_drowsy_prob is None:
                ema_drowsy_prob = drowsy_prob
            else:
                ema_drowsy_prob = EMA_ALPHA * drowsy_prob + (1 - EMA_ALPHA) * ema_drowsy_prob

            cnn_drowsy_now = ema_drowsy_prob >= CNN_DROWSY_THRESHOLD

            if cnn_drowsy_now:
                consec_cnn_drowsy += 1
            else:
                consec_cnn_drowsy = max(0, consec_cnn_drowsy - 2)

            # -------- Fusion logic --------
            # Strong event if:
            # 1) eyes have been closed long enough
            # OR
            # 2) CNN has persistent drowsy signal
            # OR
            # 3) both are moderately bad together
            eye_event = consec_eye_closed >= EYE_CLOSED_CONSEC_FRAMES
            cnn_event = consec_cnn_drowsy >= FATIGUE_CONSEC_TRIGGER
            combined_event = eye_closed_now and (ema_drowsy_prob >= 0.50)

            fusion_flag = 1 if (eye_event or cnn_event or combined_event) else 0
            fusion_window.append(fusion_flag)

            alert_ratio = sum(fusion_window) / len(fusion_window) if len(fusion_window) > 0 else 0.0
            fatigue_detected = (
                cnn_event
                or eye_event
                or (len(fusion_window) == WINDOW_SIZE and alert_ratio >= WINDOW_ALERT_RATIO)
            )

            now = time.time()
            if fatigue_detected:
                current_status = "FATIGUE ALERT"
                current_confidence = max(
                    ema_drowsy_prob if ema_drowsy_prob is not None else 0.0,
                    1.0 - min(avg_ear / EAR_CLOSED_THRESHOLD, 1.0) if avg_ear is not None else 0.0,
                    alert_ratio
                )
                last_alert_time = now
            else:
                if now - last_alert_time < ALERT_HOLD_SEC:
                    current_status = "FATIGUE ALERT"
                else:
                    if eye_closed_now or (ema_drowsy_prob is not None and ema_drowsy_prob >= 0.50):
                        current_status = "DROWSY"
                        current_confidence = max(
                            ema_drowsy_prob if ema_drowsy_prob is not None else 0.0,
                            1.0 - min(avg_ear / EAR_CLOSED_THRESHOLD, 1.0) if avg_ear is not None else 0.0,
                        )
                    else:
                        current_status = "NATURAL"
                        current_confidence = 1.0 - (ema_drowsy_prob if ema_drowsy_prob is not None else 0.0)

        else:
            current_status = "NO FACE"
            current_confidence = 0.0
            consec_eye_closed = 0
            consec_cnn_drowsy = 0
            fusion_window.clear()

    # =========================================
    # DISPLAY
    # =========================================
    if current_status == "FATIGUE ALERT":
        color = (0, 0, 255)
    elif current_status == "DROWSY":
        color = (0, 165, 255)
    elif current_status == "NO FACE":
        color = (255, 0, 0)
    else:
        color = (0, 255, 0)

    cv2.putText(
        display,
        f"{current_status} ({current_confidence:.2f})",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    if SHOW_DEBUG:
        cv2.putText(display, f"EAR: {last_avg_ear:.3f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, f"Eye closed frames: {consec_eye_closed}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, f"CNN drowsy prob: {last_cnn_drowsy_prob:.2f}", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, f"Smoothed CNN: {0.0 if ema_drowsy_prob is None else ema_drowsy_prob:.2f}", (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, f"CNN consec: {consec_cnn_drowsy}", (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, f"Fusion window: {sum(fusion_window)}/{len(fusion_window)}", (20, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Hybrid Fatigue Detection", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_landmarker.close()