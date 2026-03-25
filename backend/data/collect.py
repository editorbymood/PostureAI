"""
Data Collection Tool — Posture Annotation via Webcam
=====================================================
Collect labeled posture data using MediaPipe Pose.

Usage:
    python collect.py

Controls:
    0-4  : Set current label (see POSTURE_LABELS)
    SPACE: Toggle recording on/off
    S    : Save collected data to file
    Q    : Quit

The tool captures keypoints via MediaPipe, computes the
same 12 engineered features as the frontend, and saves
them with labels for training.
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import os
import math

# Consistent with frontend & backend
FEATURE_NAMES = [
    'shoulder_angle', 'neck_inclination', 'torso_inclination',
    'hip_angle', 'head_lateral_offset', 'left_torso_ratio',
    'right_torso_ratio', 'symmetry_score', 'head_drop_ratio',
    'ear_alignment', 'shoulder_hip_width', 'avg_confidence',
]

POSTURE_LABELS = ['Good', 'Fwd Lean', 'Left Lean', 'Right Lean', 'Neck Strain']
LABEL_COLORS = [
    (0, 200, 0), (0, 180, 255), (255, 180, 0),
    (200, 0, 255), (0, 0, 255),
]

# MediaPipe landmark indices (different from COCO!)
MP_NOSE = 0
MP_LEFT_EYE = 2
MP_RIGHT_EYE = 5
MP_LEFT_EAR = 7
MP_RIGHT_EAR = 8
MP_LEFT_SHOULDER = 11
MP_RIGHT_SHOULDER = 12
MP_LEFT_HIP = 23
MP_RIGHT_HIP = 24


def extract_features(landmarks, w, h):
    """Extract 12 features from MediaPipe landmarks."""
    def lm(idx):
        l = landmarks[idx]
        return {'x': l.x * w, 'y': l.y * h, 'v': l.visibility}

    nose = lm(MP_NOSE)
    l_shoulder = lm(MP_LEFT_SHOULDER)
    r_shoulder = lm(MP_RIGHT_SHOULDER)
    l_hip = lm(MP_LEFT_HIP)
    r_hip = lm(MP_RIGHT_HIP)
    l_ear = lm(MP_LEFT_EAR)
    r_ear = lm(MP_RIGHT_EAR)

    midS = {'x': (l_shoulder['x'] + r_shoulder['x']) / 2,
            'y': (l_shoulder['y'] + r_shoulder['y']) / 2}
    midH = {'x': (l_hip['x'] + r_hip['x']) / 2,
            'y': (l_hip['y'] + r_hip['y']) / 2}

    sw = math.dist([l_shoulder['x'], l_shoulder['y']], [r_shoulder['x'], r_shoulder['y']])
    hw = math.dist([l_hip['x'], l_hip['y']], [r_hip['x'], r_hip['y']])
    bh = math.dist([nose['x'], nose['y']], [midH['x'], midH['y']])
    if sw < 1: sw = 1
    if bh < 1: bh = sw * 2.5

    features = [0.0] * 12

    # 0: shoulder_angle
    dy = r_shoulder['y'] - l_shoulder['y']
    dx = r_shoulder['x'] - l_shoulder['x']
    features[0] = abs(math.degrees(math.atan2(dy, dx)))
    if features[0] > 90: features[0] = 180 - features[0]

    # 1: neck_inclination
    nx = nose['x'] - midS['x']
    ny = midS['y'] - nose['y']
    features[1] = abs(math.degrees(math.atan2(nx, ny)))

    # 2: torso_inclination
    tx = midS['x'] - midH['x']
    ty = midH['y'] - midS['y']
    features[2] = abs(math.degrees(math.atan2(tx, ty)))

    # 3: hip_angle
    hdy = r_hip['y'] - l_hip['y']
    hdx = r_hip['x'] - l_hip['x']
    features[3] = abs(math.degrees(math.atan2(hdy, hdx)))
    if features[3] > 90: features[3] = 180 - features[3]

    # 4: head_lateral_offset
    features[4] = (nose['x'] - midS['x']) / sw

    # 5, 6: torso ratios
    features[5] = math.dist([l_shoulder['x'], l_shoulder['y']], [l_hip['x'], l_hip['y']]) / bh
    features[6] = math.dist([r_shoulder['x'], r_shoulder['y']], [r_hip['x'], r_hip['y']]) / bh

    # 7: symmetry
    pairs = [(l_shoulder, r_shoulder), (l_hip, r_hip)]
    total_diff = 0
    for lp, rp in pairs:
        ld = abs(lp['x'] - midS['x'])
        rd = abs(rp['x'] - midS['x'])
        mx = max(ld, rd, 1)
        total_diff += abs(ld - rd) / mx
    features[7] = max(0, 1 - total_diff / len(pairs))

    # 8: head_drop_ratio
    expected_y = midS['y'] - sw * 0.8
    features[8] = (nose['y'] - expected_y) / bh

    # 9: ear_alignment
    ear = l_ear if l_ear['v'] > r_ear['v'] else r_ear
    sh = l_shoulder if l_ear['v'] > r_ear['v'] else r_shoulder
    ex = ear['x'] - sh['x']
    ey = sh['y'] - ear['y']
    features[9] = abs(math.degrees(math.atan2(ex, ey)))

    # 10: shoulder_hip_width
    features[10] = sw / hw if hw > 0 else 1.0

    # 11: avg_confidence
    visibilities = [landmarks[i].visibility for i in range(33)]
    features[11] = sum(visibilities) / len(visibilities)

    return features


def main():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)

    current_label = 0
    is_recording = False
    collected_data = []
    frame_count = 0

    print("\n" + "=" * 50)
    print("  PostureAI Data Collection Tool")
    print("=" * 50)
    print("  Keys: 0-4 = set label | SPACE = record | S = save | Q = quit")
    print("  Labels:", {i: l for i, l in enumerate(POSTURE_LABELS)})
    print()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            features = extract_features(results.pose_landmarks.landmark, w, h)

            if is_recording:
                collected_data.append({
                    'features': features,
                    'label': current_label,
                })
                frame_count += 1

            # Draw landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # UI overlay
        color = LABEL_COLORS[current_label]
        status = "● RECORDING" if is_recording else "○ PAUSED"
        cv2.putText(frame, f"Label: {current_label} - {POSTURE_LABELS[current_label]}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, status, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255) if is_recording else (128, 128, 128), 2)
        cv2.putText(frame, f"Collected: {len(collected_data)} frames",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow('PostureAI Data Collection', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            is_recording = not is_recording
            print(f"  {'▶ Recording' if is_recording else '⏸ Paused'} | Label: {POSTURE_LABELS[current_label]}")
        elif key in [ord(str(i)) for i in range(5)]:
            current_label = int(chr(key))
            print(f"  Label set to: {current_label} ({POSTURE_LABELS[current_label]})")
        elif key == ord('s') and len(collected_data) > 0:
            save_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
            os.makedirs(save_dir, exist_ok=True)
            filepath = os.path.join(save_dir, f"posture_data_{int(time.time())}.npy")
            np.save(filepath, collected_data, allow_pickle=True)
            print(f"  💾 Saved {len(collected_data)} samples to {filepath}")

    cap.release()
    cv2.destroyAllWindows()
    pose.close()

    if len(collected_data) > 0:
        save_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f"posture_data_{int(time.time())}.npy")
        np.save(filepath, collected_data, allow_pickle=True)
        print(f"\n✅ Auto-saved {len(collected_data)} samples to {filepath}")


if __name__ == "__main__":
    main()
