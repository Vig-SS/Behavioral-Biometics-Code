# run_behavior_classifier.py  (paper version)
# Siddhi Satheesh — CSCI 693
# Project: Behavioral Biometrics for Distinguishing Visually Similar Users
#
# Live demo classifier, importing shared pose_common. Functionally the same as
# the original: rolling window -> features -> smoothed prediction on screen.
#
# Usage:  python run_behavior_classifier.py --model behavior_classifier.joblib

import argparse
from collections import deque, Counter

import cv2
import joblib
import numpy as np
import mediapipe as mp

from pose_common import (
    normalize_selected_pose_landmarks, extract_window_features_from_array,
    make_pose_estimator,
)

mp_pose = mp.solutions.pose


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="behavior_classifier.joblib")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--smooth", type=int, default=10)
    args = p.parse_args()

    saved = joblib.load(args.model)
    clf = saved["model"]
    window_size = saved["window_size"]
    expected = saved.get("expected_input_features")
    per_frame = saved.get("input_frame_features")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Error: Could not open webcam."); return

    frame_buffer = deque(maxlen=window_size)
    pred_buffer = deque(maxlen=args.smooth)
    print("Running live classifier. Press q to quit.")

    with make_pose_estimator(mp_pose) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            pose_data = normalize_selected_pose_landmarks(result.pose_landmarks)
            if per_frame and len(pose_data) != per_frame:
                print("Feature mismatch; retrain with matching scripts."); break
            frame_buffer.append(pose_data)

            label = "Collecting pose frames..."
            if len(frame_buffer) == window_size:
                feats = extract_window_features_from_array(
                    list(frame_buffer)).reshape(1, -1)
                if expected and feats.shape[1] != expected:
                    print("Feature mismatch; retrain."); break
                pred = clf.predict(feats)[0]
                pred_buffer.append(pred)
                smoothed = Counter(pred_buffer).most_common(1)[0][0]
                label = f"Predicted user: {smoothed}"
                if hasattr(clf, "predict_proba"):
                    conf = float(np.max(clf.predict_proba(feats)[0]))
                    label += f"  ({conf:.2f})"

            cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_COMPLEX,
                        0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Frames: {len(frame_buffer)}/{window_size}",
                        (20, 80), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (255, 255, 255), 2)
            cv2.imshow("Live Behavior Classifier", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
