import cv2
import numpy as np
import tensorflow as tf
import time
import math
from collections import deque

import mediapipe as mp
from mediapipe.tasks.python import vision

# =========================================
# SETTINGS
# =========================================
CNN_MODEL_PATH = "model_1_simple_cnn.keras"
FACE_LANDMARKER_MODEL_PATH = r"C:\Users\User\Desktop\fatigue\face_landmarker.task"

IMG_HEIGHT = 128
IMG_WIDTH = 128
CLASS_NAMES = ['DROWSY', 'NATURAL']  # index 0 = DROWSY, index 1 = NATURAL

# -----------------------------------------
# Calibration
# -----------------------------------------
CALIBRATION_SECONDS = 3.0
CALIBRATION_MAX_WAIT = 12.0

# EAR relative threshold
EAR_DROP_RATIO = 0.75
EAR_THRESH_FLOOR = 0.15
EYE_PERCLOS_WINDOW_SEC = 2.0
EYE_PERCLOS_THRESHOLD = 0.40

# MAR relative threshold
MAR_RISE_RATIO = 1.60
MAR_THRESH_FLOOR = 0.35
YAWN_MIN_DURATION_SEC = 0.8
YAWN_COOLDOWN_SEC = 2.0

# Head pose / nodding
HEAD_DOWN_PITCH_DELTA = 12.0
HEAD_DOWN_MIN_DURATION_SEC = 0.6
HEAD_DOWN_STRONG_DELTA = 18.0
HEAD_DOWN_STRONG_MIN_SEC = 1.0
HEAD_DOWN_FORCE_ALERT_SEC = 3.0

NOD_MIN_DROP_DEG = 8.0
NOD_RETURN_DEG = 5.0
NOD_MAX_CYCLE_SEC = 1.5
NOD_COUNT_WINDOW_SEC = 4.0
NOD_COUNT_TRIGGER = 2

# CNN
CNN_INFER_EVERY_N = 5
EMA_ALPHA = 0.30
PITCH_EMA_ALPHA = 0.25
CNN_DROWSY_THRESHOLD = 0.65
CNN_MIN_DURATION_SEC = 1.2

# Fusion / alert state
FRAME_SKIP = 2
FUSION_WINDOW_SEC = 3.0
FUSION_ALERT_RATIO = 0.60
FUSION_CLEAR_RATIO = 0.30
ALERT_HOLD_SEC = 2.0
ALERT_ENTER_SEC = 1.2
ALERT_EXIT_SEC = 1.8
POST_CALIB_WARMUP_SEC = 1.5
NO_FACE_GRACE_SEC = 1.0
DROWSY_SCORE_THRESHOLD = 3

# Max possible fusion_score = 3+2+2+2+2+1+1 = 13
MAX_FUSION_SCORE = 13

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
print("CNN loaded.")
print("Face Landmarker loaded.")

# =========================================
# LANDMARK INDEXES
# =========================================
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

MOUTH = {
    "left_corner": 61,
    "right_corner": 291,
    "top1": 13,
    "bottom1": 14,
    "top2": 81,
    "bottom2": 178,
}

POSE_IDX = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}

# =========================================
# HELPERS
# =========================================
def euclidean(p1, p2):
    return math.dist(p1, p2)

def lm_to_px(lm, w, h):
    return (lm.x * w, lm.y * h)

def compute_eye_ear(face_landmarks, eye_def, w, h):
    pts = {name: lm_to_px(face_landmarks[idx], w, h) for name, idx in eye_def.items()}
    horizontal = euclidean(pts["left_corner"], pts["right_corner"])
    vertical1 = euclidean(pts["top1"], pts["bottom1"])
    vertical2 = euclidean(pts["top2"], pts["bottom2"])
    if horizontal == 0:
        return None
    return (vertical1 + vertical2) / (2.0 * horizontal)

def compute_avg_ear(face_landmarks, w, h):
    left_ear = compute_eye_ear(face_landmarks, LEFT_EYE, w, h)
    right_ear = compute_eye_ear(face_landmarks, RIGHT_EYE, w, h)
    if left_ear is None or right_ear is None:
        return None
    return (left_ear + right_ear) / 2.0

def compute_mar(face_landmarks, w, h):
    pts = {name: lm_to_px(face_landmarks[idx], w, h) for name, idx in MOUTH.items()}
    horizontal = euclidean(pts["left_corner"], pts["right_corner"])
    vertical1 = euclidean(pts["top1"], pts["bottom1"])
    vertical2 = euclidean(pts["top2"], pts["bottom2"])
    if horizontal == 0:
        return None
    return (vertical1 + vertical2) / (2.0 * horizontal)

def crop_face_from_landmarks(frame, face_landmarks):
    h, w = frame.shape[:2]
    xs = [lm.x for lm in face_landmarks]
    ys = [lm.y for lm in face_landmarks]

    x_min = max(0, int(min(xs) * w) - 20)
    y_min = max(0, int(min(ys) * h) - 20)
    x_max = min(w, int(max(xs) * w) + 20)
    y_max = min(h, int(max(ys) * h) + 20)

    if x_max <= x_min or y_max <= y_min:
        return None

    crop = frame[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return None
    return crop

def preprocess_for_cnn(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_WIDTH, IMG_HEIGHT))
    img_array = resized.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def get_head_pitch_deg(face_landmarks, frame_w, frame_h):
    image_points = np.array([
        lm_to_px(face_landmarks[POSE_IDX["nose_tip"]], frame_w, frame_h),
        lm_to_px(face_landmarks[POSE_IDX["chin"]], frame_w, frame_h),
        lm_to_px(face_landmarks[POSE_IDX["left_eye_outer"]], frame_w, frame_h),
        lm_to_px(face_landmarks[POSE_IDX["right_eye_outer"]], frame_w, frame_h),
        lm_to_px(face_landmarks[POSE_IDX["left_mouth"]], frame_w, frame_h),
        lm_to_px(face_landmarks[POSE_IDX["right_mouth"]], frame_w, frame_h),
    ], dtype=np.float64)

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1),
    ], dtype=np.float64)

    focal_length = frame_w
    center = (frame_w / 2, frame_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))
    success, rotation_vec, _ = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    proj_mat = np.hstack((rotation_mat, np.zeros((3, 1))))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_mat)
    euler_angles = euler_angles.flatten()
    pitch = float(euler_angles[0])
    return pitch

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def update_binary_time_counter(active, current_value, dt, decay_rate=1.5):
    if active:
        return current_value + dt
    return max(0.0, current_value - decay_rate * dt)

def trim_deque_by_age(dq, now_ts, max_age_sec):
    while dq and (now_ts - dq[0][0]) > max_age_sec:
        dq.popleft()
# =========================================
# VIDEO
# =========================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam.")
    raise SystemExit

# Get actual camera FPS for accurate MediaPipe timestamps
cam_fps = cap.get(cv2.CAP_PROP_FPS)
if cam_fps <= 0 or cam_fps > 120:
    cam_fps = 30.0  # safe fallback

print(f"Camera FPS: {cam_fps:.1f}")
print("Press 'q' to quit.")
processed_dt = FRAME_SKIP / cam_fps
eye_buffer_maxlen = max(1, int(EYE_PERCLOS_WINDOW_SEC / processed_dt))
fusion_history_maxlen = max(1, int(FUSION_WINDOW_SEC / processed_dt))

frame_index = 0
detection_frame_count = 0  # counts frames that actually ran detection

# -----------------------------------------
# Calibration state
# -----------------------------------------
calibration_start = time.time()
ear_samples = []
mar_samples = []
pitch_samples = []
calibrated = False
calibration_quality = "none"  # "full" | "partial" | "fallback" | "hardcoded"
calibrated_at = None

baseline_ear = None
baseline_mar = None
baseline_pitch = None
pitch_calibrated = False
dyn_ear_thresh = None
dyn_mar_thresh = None

# -----------------------------------------
# Detection state
# -----------------------------------------
ema_drowsy_prob = None
smoothed_pitch = None

eye_closed_buffer = deque()
eye_event = False

yawn_active_sec = 0.0
head_down_active_sec = 0.0
head_down_force_sec = 0.0
cnn_high_sec = 0.0

pitch_history = deque()
nod_events = deque()
nod_phase = "neutral"   # "neutral" | "dropping"
nod_start_time = None
nod_peak_pitch = None

fusion_history = deque()

last_yawn_time = 0.0
last_alert_time = 0.0
last_face_seen_time = 0.0

last_ear = 0.0
last_mar = 0.0
last_pitch = 0.0
current_status = "INITIALIZING"
current_confidence = 0.0
yawn_count = 0
perclos = 0.0

fusion_score = 0
yawn_event = False
head_down_event = False
nodding_event = False
cnn_event = False
mouth_wide_now = False
eye_closed_now = False
alert_entry_sec = 0.0
alert_clear_sec = 0.0

# =========================================
# MAIN LOOP
# =========================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame.")
        break

    frame = cv2.flip(frame, 1)
    display = frame.copy()
    frame_index += 1
    h, w = frame.shape[:2]

    # Use frame-index-based timestamp for MediaPipe VIDEO mode
    # This guarantees monotonically increasing values regardless of processing lag
    timestamp_ms = int(frame_index * (1000.0 / cam_fps))

    if frame_index % FRAME_SKIP != 0:
        # Skip this frame — still render display from last known state
        pass
    else:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = face_landmarker.detect_for_video(mp_image, timestamp_ms)

        face_found = result.face_landmarks is not None and len(result.face_landmarks) > 0

        if face_found:
            last_face_seen_time = time.time()
            face_landmarks = result.face_landmarks[0]

            avg_ear = compute_avg_ear(face_landmarks, w, h)
            mar = compute_mar(face_landmarks, w, h)
            pitch = get_head_pitch_deg(face_landmarks, w, h)

            if avg_ear is not None:
                last_ear = avg_ear
            if mar is not None:
                last_mar = mar
            if pitch is not None:
                last_pitch = pitch

            # =====================================================
            # CALIBRATION BLOCK
            # =====================================================
            if not calibrated:
                current_status = "CALIBRATING - LOOK NORMAL"
                elapsed = time.time() - calibration_start

                # Stability gates — only accept samples when face is in a
                # "clearly normal and stable" state to avoid contaminating
                # the baseline with blinks, yawns, or head movement.
                eyes_clearly_open = avg_ear is not None and avg_ear > 0.20
                mouth_clearly_closed = mar is not None and mar < 0.35
                pitch_stable = (
                    pitch is not None
                    and (
                        len(pitch_samples) == 0
                        or abs(pitch - pitch_samples[-1]) < 3.0
                    )
                )

                if avg_ear is not None and 0.12 <= avg_ear <= 0.50 and eyes_clearly_open:
                    ear_samples.append(avg_ear)

                if mar is not None and 0.05 <= mar <= 0.90 and mouth_clearly_closed:
                    mar_samples.append(mar)

                # Pitch is optional — collect if stable, but don't require it
                if pitch is not None and -45 <= pitch <= 45 and pitch_stable:
                    pitch_samples.append(pitch)

                enough_ear = len(ear_samples) >= 5
                enough_mar = len(mar_samples) >= 5

                if elapsed >= CALIBRATION_SECONDS and enough_ear and enough_mar:
                    # ---- Normal successful calibration ----
                    baseline_ear = float(np.median(ear_samples))
                    baseline_mar = float(np.median(mar_samples))
                    pitch_calibrated = len(pitch_samples) >= 3
                    baseline_pitch = float(np.median(pitch_samples)) if pitch_calibrated else None

                    dyn_ear_thresh = max(EAR_THRESH_FLOOR, baseline_ear * EAR_DROP_RATIO)
                    dyn_mar_thresh = max(MAR_THRESH_FLOOR, baseline_mar * MAR_RISE_RATIO)

                    calibration_quality = "full"
                    calibrated = True
                    calibrated_at = time.time()

                    print(f"[CALIB] Quality: {calibration_quality}")
                    print(f"[CALIB] Baseline EAR:   {baseline_ear:.4f}")
                    print(f"[CALIB] Baseline MAR:   {baseline_mar:.4f}")
                    if baseline_pitch is not None:
                        print(f"[CALIB] Baseline Pitch: {baseline_pitch:.2f} deg")
                    else:
                        print("[CALIB] Baseline Pitch: N/A (pitch not calibrated)")
                    print(f"[CALIB] EAR threshold:  {dyn_ear_thresh:.4f}")
                    print(f"[CALIB] MAR threshold:  {dyn_mar_thresh:.4f}")
                    print(f"[CALIB] EAR samples: {len(ear_samples)}, MAR samples: {len(mar_samples)}, Pitch samples: {len(pitch_samples)}")

                elif elapsed >= CALIBRATION_MAX_WAIT:
                    # ---- Hard fallback — never restart-loop past this point ----
                    # Use whatever samples we collected; fall back to population
                    # defaults only if we have nothing at all.
                    if ear_samples and mar_samples:
                        baseline_ear = float(np.median(ear_samples))
                        baseline_mar = float(np.median(mar_samples))
                        calibration_quality = "partial"
                    else:
                        baseline_ear = 0.25
                        baseline_mar = 0.20
                        calibration_quality = "hardcoded"

                    pitch_calibrated = len(pitch_samples) >= 3
                    baseline_pitch = float(np.median(pitch_samples)) if pitch_calibrated else None
                    dyn_ear_thresh = max(EAR_THRESH_FLOOR, baseline_ear * EAR_DROP_RATIO)
                    dyn_mar_thresh = max(MAR_THRESH_FLOOR, baseline_mar * MAR_RISE_RATIO)
                    calibrated = True
                    calibrated_at = time.time()

                    print(f"[WARN] Forced calibration fallback. Quality: {calibration_quality}")
                    print(f"[CALIB] Baseline EAR:   {baseline_ear:.4f}")
                    print(f"[CALIB] Baseline MAR:   {baseline_mar:.4f}")
                    if baseline_pitch is not None:
                        print(f"[CALIB] Baseline Pitch: {baseline_pitch:.2f} deg")
                    else:
                        print("[CALIB] Baseline Pitch: N/A (pitch not calibrated)")
                    print(f"[CALIB] EAR threshold:  {dyn_ear_thresh:.4f}")
                    print(f"[CALIB] MAR threshold:  {dyn_mar_thresh:.4f}")
                    print(f"[CALIB] EAR samples: {len(ear_samples)}, MAR samples: {len(mar_samples)}, Pitch samples: {len(pitch_samples)}")

                # Show live calibration progress on-screen
                total_needed = CALIBRATION_SECONDS
                progress = min(elapsed / total_needed, 1.0) if total_needed > 0 else 1.0
                bar_width = 200
                filled = int(bar_width * progress)
                cv2.rectangle(display, (20, h - 50), (20 + bar_width, h - 30), (80, 80, 80), -1)
                cv2.rectangle(display, (20, h - 50), (20 + filled, h - 30), (0, 220, 0), -1)
                cv2.putText(display, f"EAR samples: {len(ear_samples)}  MAR samples: {len(mar_samples)}",
                            (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # =====================================================
            # DETECTION BLOCK
            # =====================================================
            else:
                detection_frame_count += 1
                now = time.time()

                # ---- Smooth pitch before using it ----
                if pitch is not None:
                    if smoothed_pitch is None:
                        smoothed_pitch = pitch
                    else:
                        smoothed_pitch = PITCH_EMA_ALPHA * pitch + (1.0 - PITCH_EMA_ALPHA) * smoothed_pitch
                    last_pitch = smoothed_pitch

                # ---- 1) Eye closure: time-normalized PERCLOS ----
                eye_closed_now = avg_ear is not None and dyn_ear_thresh is not None and avg_ear < dyn_ear_thresh
                eye_closed_buffer.append(1 if eye_closed_now else 0)
                while len(eye_closed_buffer) > eye_buffer_maxlen:
                    eye_closed_buffer.popleft()

                perclos = sum(eye_closed_buffer) / len(eye_closed_buffer) if eye_closed_buffer else 0.0
                eye_buffer_ready = len(eye_closed_buffer) >= max(5, int(0.5 * eye_buffer_maxlen))
                eye_event = eye_buffer_ready and (perclos >= EYE_PERCLOS_THRESHOLD)

                # ---- 2) Yawning with time-based persistence ----
                mouth_wide_now = mar is not None and dyn_mar_thresh is not None and mar > dyn_mar_thresh
                yawn_active_sec = update_binary_time_counter(mouth_wide_now, yawn_active_sec, processed_dt, decay_rate=2.0)

                yawn_event = False
                if yawn_active_sec >= YAWN_MIN_DURATION_SEC and (now - last_yawn_time) > YAWN_COOLDOWN_SEC:
                    yawn_event = True
                    yawn_count += 1
                    last_yawn_time = now
                    yawn_active_sec = 0.0

                # ---- 3) Head down with smoothed pitch ----
                head_down_now = False
                pitch_delta = None
                if pitch_calibrated and smoothed_pitch is not None and baseline_pitch is not None:
                    pitch_delta = smoothed_pitch - baseline_pitch
                    # Primary directional check (current setup sign)
                    head_down_now = pitch_delta > HEAD_DOWN_PITCH_DELTA
                    # Fallback for opposite-sign camera setups: require much stronger tilt
                    if not head_down_now:
                        head_down_now = pitch_delta < -HEAD_DOWN_STRONG_DELTA

                head_down_active_sec = update_binary_time_counter(head_down_now, head_down_active_sec, processed_dt, decay_rate=1.0)
                head_down_event = head_down_active_sec >= HEAD_DOWN_MIN_DURATION_SEC
                head_down_strong_event = head_down_active_sec >= HEAD_DOWN_STRONG_MIN_SEC
                head_down_force_sec = update_binary_time_counter(head_down_now, head_down_force_sec, processed_dt, decay_rate=0.5)
                head_down_force_alert = head_down_force_sec >= HEAD_DOWN_FORCE_ALERT_SEC

                # ---- 4) Nod detection: down then return up, time-bounded ----
                if pitch_calibrated and smoothed_pitch is not None and baseline_pitch is not None:
                    if nod_phase == "neutral":
                        if smoothed_pitch > (baseline_pitch + NOD_MIN_DROP_DEG):
                            nod_phase = "dropping"
                            nod_start_time = now
                            nod_peak_pitch = smoothed_pitch

                    elif nod_phase == "dropping":
                        nod_peak_pitch = max(nod_peak_pitch, smoothed_pitch)

                        returned_up = (nod_peak_pitch - smoothed_pitch) >= NOD_RETURN_DEG
                        timed_out = (now - nod_start_time) > NOD_MAX_CYCLE_SEC

                        if returned_up:
                            nod_events.append((now, 1))
                            nod_phase = "neutral"
                            nod_start_time = None
                            nod_peak_pitch = None
                        elif timed_out:
                            nod_phase = "neutral"
                            nod_start_time = None
                            nod_peak_pitch = None

                trim_deque_by_age(nod_events, now, NOD_COUNT_WINDOW_SEC)
                nodding_event = sum(v for _, v in nod_events) >= NOD_COUNT_TRIGGER

                # ---- 5) CNN inference (throttled) ----
                if detection_frame_count % CNN_INFER_EVERY_N == 0:
                    face_crop = crop_face_from_landmarks(frame, face_landmarks)
                    if face_crop is not None:
                        cnn_input = preprocess_for_cnn(face_crop)
                        pred = cnn_model(tf.constant(cnn_input), training=False).numpy()

                        if pred.shape[-1] == 1:
                            raw_pred = float(pred[0][0])

                            # ASSUMPTION:
                            # sigmoid output = P(DROWSY)
                            drowsy_prob = raw_pred

                            # If your model instead outputs P(NATURAL), replace with:
                            # drowsy_prob = 1.0 - raw_pred
                        else:
                            # Assumes softmax order [DROWSY, NATURAL]
                            drowsy_prob = float(pred[0][0])

                        if ema_drowsy_prob is None:
                            ema_drowsy_prob = drowsy_prob
                        else:
                            ema_drowsy_prob = EMA_ALPHA * drowsy_prob + (1.0 - EMA_ALPHA) * ema_drowsy_prob

                cnn_drowsy_now = ema_drowsy_prob is not None and ema_drowsy_prob >= CNN_DROWSY_THRESHOLD
                cnn_high_sec = update_binary_time_counter(cnn_drowsy_now, cnn_high_sec, processed_dt, decay_rate=2.0)
                cnn_event = cnn_high_sec >= CNN_MIN_DURATION_SEC
                # Prevent single noisy channels from dominating fusion.
                # CNN-only evidence should support other cues, not drive the state alone.
                cnn_for_fusion = cnn_event and (eye_event or mouth_wide_now or head_down_event or nodding_event)
                head_down_for_fusion = head_down_event and (eye_event or cnn_drowsy_now or mouth_wide_now)

                # =====================================================
                # FUSION
                # =====================================================
                fusion_score = 0

                if eye_event:
                    fusion_score += 3
                if yawn_event:
                    fusion_score += 2
                if head_down_for_fusion:
                    fusion_score += 2
                elif head_down_strong_event:
                    # Strong sustained head-down alone can indicate fatigue.
                    fusion_score += 2
                elif head_down_event:
                    fusion_score += 1
                if nodding_event:
                    fusion_score += 2
                if cnn_for_fusion:
                    fusion_score += 2
                if ema_drowsy_prob is not None and ema_drowsy_prob >= 0.50 and (eye_event or mouth_wide_now):
                    fusion_score += 1
                if mouth_wide_now and eye_closed_now:
                    fusion_score += 1

                alert_threshold = 5 if calibration_quality in ("partial", "hardcoded") else 4

                fatigue_flag = 1 if fusion_score >= alert_threshold else 0
                fusion_history.append((now, fatigue_flag))
                trim_deque_by_age(fusion_history, now, FUSION_WINDOW_SEC)

                fusion_ratio_alert = False
                fusion_ratio = 0.0
                if fusion_history:
                    fusion_ratio = sum(v for _, v in fusion_history) / len(fusion_history)
                    fusion_window_ready = len(fusion_history) >= max(5, int(0.6 * fusion_history_maxlen))
                    fusion_ratio_alert = fusion_window_ready and (fusion_ratio >= FUSION_ALERT_RATIO)

                current_confidence = min(1.0, fusion_score / MAX_FUSION_SCORE)
                # Timed alert state machine:
                # - Enter ALERT only after sustained fusion-window evidence
                # - Exit ALERT only after sustained low-fatigue evidence
                if fusion_ratio_alert:
                    alert_entry_sec += processed_dt
                else:
                    alert_entry_sec = 0.0

                # Use a separate, lower clear threshold to avoid ALERT latch.
                low_fatigue_now = (fusion_ratio <= FUSION_CLEAR_RATIO) and (fusion_score <= 1)
                if low_fatigue_now:
                    alert_clear_sec += processed_dt
                else:
                    alert_clear_sec = 0.0

                in_post_calib_warmup = (
                    calibrated_at is not None
                    and (now - calibrated_at) < POST_CALIB_WARMUP_SEC
                )

                if in_post_calib_warmup:
                    current_status = "NATURAL"
                    current_confidence = 1.0 - (ema_drowsy_prob if ema_drowsy_prob is not None else 0.0)
                    alert_entry_sec = 0.0
                    alert_clear_sec = 0.0
                elif head_down_force_alert:
                    # Safety rule: sustained head-down alone triggers alert.
                    current_status = "FATIGUE ALERT"
                    current_confidence = min(1.0, max(current_confidence, 0.80))
                    last_alert_time = now
                    alert_entry_sec = ALERT_ENTER_SEC
                    alert_clear_sec = 0.0
                elif current_status != "FATIGUE ALERT":
                    if alert_entry_sec >= ALERT_ENTER_SEC:
                        current_status = "FATIGUE ALERT"
                        last_alert_time = now
                        alert_clear_sec = 0.0
                    elif fusion_score >= DROWSY_SCORE_THRESHOLD:
                        current_status = "DROWSY"
                    else:
                        current_status = "NATURAL"
                        current_confidence = 1.0 - (ema_drowsy_prob if ema_drowsy_prob is not None else 0.0)
                else:
                    can_release_alert = (
                        (now - last_alert_time) >= ALERT_HOLD_SEC
                        and alert_clear_sec >= ALERT_EXIT_SEC
                    )
                    if can_release_alert:
                        if fusion_score >= DROWSY_SCORE_THRESHOLD:
                            current_status = "DROWSY"
                        else:
                            current_status = "NATURAL"
                            current_confidence = 1.0 - (ema_drowsy_prob if ema_drowsy_prob is not None else 0.0)
                        alert_entry_sec = 0.0
                    else:
                        current_status = "FATIGUE ALERT"
        else:
            # Brief face-loss is common during downward nods.
            # Keep previous state for a short grace period.
            no_face_duration = time.time() - last_face_seen_time
            if no_face_duration < NO_FACE_GRACE_SEC:
                pass
            else:
                current_status = "NO FACE"
                current_confidence = 0.0

                eye_closed_buffer.clear()
                eye_event = False

                yawn_active_sec = 0.0
                head_down_active_sec = 0.0
                head_down_force_sec = 0.0
                cnn_high_sec = 0.0

                yawn_event = False
                head_down_event = False
                nodding_event = False
                cnn_event = False
                mouth_wide_now = False
                eye_closed_now = False

                ema_drowsy_prob = None
                smoothed_pitch = None

                pitch_history.clear()
                nod_events.clear()
                nod_phase = "neutral"
                nod_start_time = None
                nod_peak_pitch = None

                fusion_history.clear()
                fusion_score = 0
                perclos = 0.0
                alert_entry_sec = 0.0
                alert_clear_sec = 0.0

    # =====================================================
    # DISPLAY
    # =====================================================
    if current_status == "FATIGUE ALERT":
        color = (0, 0, 255)
    elif current_status == "DROWSY":
        color = (0, 165, 255)
    elif current_status in ["NO FACE", "INITIALIZING", "CALIBRATING - LOOK NORMAL"]:
        color = (255, 0, 0)
    else:
        color = (0, 255, 0)

    cv2.putText(display, f"{current_status}  score:{current_confidence:.2f}",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    if calibrated and calibration_quality != "full":
        cv2.putText(display, f"CALIB: {calibration_quality.upper()}",
                    (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

    if SHOW_DEBUG:
        cv2.putText(display, f"EAR: {last_ear:.3f}  thresh: {dyn_ear_thresh:.3f}" if dyn_ear_thresh else f"EAR: {last_ear:.3f}",
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(display, f"MAR: {last_mar:.3f}  thresh: {dyn_mar_thresh:.3f}" if dyn_mar_thresh else f"MAR: {last_mar:.3f}",
                    (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(display, f"Pitch: {last_pitch:.2f}  baseline: {baseline_pitch:.2f}" if baseline_pitch is not None else f"Pitch: {last_pitch:.2f}",
                    (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(display, f"PERCLOS: {perclos:.2f}  (eye_event: {eye_event if calibrated else 'N/A'})",
                    (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(display, f"Yawn active: {yawn_active_sec:.2f}s  count: {yawn_count}",
                    (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(display, f"Head down active: {head_down_active_sec:.2f}s",
                    (20, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(display, f"Nod phase: {nod_phase}  nods in window: {sum(v for _, v in nod_events)}",
            (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(display, f"CNN EMA drowsy: {0.0 if ema_drowsy_prob is None else ema_drowsy_prob:.2f}  high: {cnn_high_sec:.2f}s",
                    (20, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(display, f"Fusion score: {fusion_score if calibrated else 'N/A'} / {MAX_FUSION_SCORE}",
                    (20, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    cv2.imshow("Fatigue Detection", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_landmarker.close()