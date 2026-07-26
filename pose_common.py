# pose_common.py
# Siddhi Satheesh — CSCI 693
# Project: Behavioral Biometrics for Distinguishing Visually Similar Users
#
# Single source of truth for the landmark set, normalization, and window
# features. capture, train, run, and every experiment script imports from here
# so the feature definition can NEVER drift between capture time and test time.
# (This is the same landmark set and normalization used in the original
# capture_behavior_data.py / train_behavior_classifier.py scripts.)

import numpy as np

# Upper-body landmarks only. Hips included for stand-up/sit-down.
# Each point stores x, y, z, visibility -> 13 landmarks * 4 = 52 values/frame.
POSE_LANDMARKS_USED = {
    0: "nose",
    2: "left_eye",
    5: "right_eye",
    7: "left_ear",
    8: "right_ear",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
}

METADATA_COLUMNS = {"timestamp", "user", "recording_id"}

NUM_POINTS = len(POSE_LANDMARKS_USED)          # 13
FRAME_FEATURES = NUM_POINTS * 4                # 52 values per frame
WINDOW_FEATURE_GROUPS = 4                      # std, motion_mean, motion_std, motion_max
WINDOW_FEATURES = FRAME_FEATURES * WINDOW_FEATURE_GROUPS   # 208


def csv_header():
    """Column header matching capture output, in a fixed order."""
    header = ["timestamp", "user", "recording_id"]
    for idx, name in POSE_LANDMARKS_USED.items():
        header += [f"{name}_{idx}_x", f"{name}_{idx}_y",
                   f"{name}_{idx}_z", f"{name}_{idx}_vis"]
    return header


def feature_column_names():
    """Just the 52 per-frame feature columns (no metadata)."""
    return csv_header()[3:]


def normalize_selected_pose_landmarks(pose_landmarks):
    """
    Center on the shoulder midpoint and scale by shoulder width.
    Removes absolute body size / camera distance / position in frame, while
    preserving posture, leaning, head tilt, shoulder and arm motion over time.
    Returns a flat list of FRAME_FEATURES floats. If no landmarks: zeros.
    """
    if pose_landmarks is None:
        return [0.0] * FRAME_FEATURES

    landmarks = pose_landmarks.landmark
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]

    center_x = (left_shoulder.x + right_shoulder.x) / 2.0
    center_y = (left_shoulder.y + right_shoulder.y) / 2.0
    center_z = (left_shoulder.z + right_shoulder.z) / 2.0

    shoulder_width = np.sqrt(
        (left_shoulder.x - right_shoulder.x) ** 2 +
        (left_shoulder.y - right_shoulder.y) ** 2 +
        (left_shoulder.z - right_shoulder.z) ** 2
    )
    if shoulder_width < 1e-6:
        shoulder_width = 1.0

    data = []
    for idx in POSE_LANDMARKS_USED.keys():
        lm = landmarks[idx]
        visibility = getattr(lm, "visibility", 0.0)
        data.extend([
            (lm.x - center_x) / shoulder_width,
            (lm.y - center_y) / shoulder_width,
            (lm.z - center_z) / shoulder_width,
            visibility,
        ])
    return data


def extract_window_features_from_array(values):
    """
    Convert a window of per-frame pose vectors (list/array shaped
    [n_frames, FRAME_FEATURES]) into a single motion-focused feature vector.
    Identical math to the original train/run scripts:
        std over time, mean |diff|, std of diff, max |diff|.
    """
    values = np.asarray(values, dtype=np.float32)
    std_features = np.std(values, axis=0)

    if len(values) > 1:
        diffs = np.diff(values, axis=0)
        motion_mean = np.mean(np.abs(diffs), axis=0)
        motion_std = np.std(diffs, axis=0)
        motion_max = np.max(np.abs(diffs), axis=0)
    else:
        motion_mean = np.zeros(values.shape[1], dtype=np.float32)
        motion_std = np.zeros(values.shape[1], dtype=np.float32)
        motion_max = np.zeros(values.shape[1], dtype=np.float32)

    return np.concatenate([std_features, motion_mean, motion_std, motion_max])


def make_pose_estimator(mp_pose, model_complexity=1):
    """Standard MediaPipe Pose config used across all scripts."""
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
