# visualize_pose_points.py
# Siddhi Satheesh — CSCI 693
# Project: Behavioral Biometrics for Distinguishing Visually Similar Users
#
import cv2
import argparse
import mediapipe as mp


mp_pose = mp.solutions.pose


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


SKELETON_CONNECTIONS = [
    (7, 2),
    (2, 0),
    (0, 5),
    (5, 8),

    (11, 12),

    (11, 13),
    (13, 15),

    (12, 14),
    (14, 16),

    (11, 23),
    (12, 24),
    (23, 24),
]


def draw_selected_skeleton(frame, pose_landmarks, show_labels=True):
    if pose_landmarks is None:
        return frame

    h, w, _ = frame.shape
    landmarks = pose_landmarks.landmark

    points = {}

    for idx, name in POSE_LANDMARKS_USED.items():
        lm = landmarks[idx]

        x = int(lm.x * w)
        y = int(lm.y * h)
        visibility = getattr(lm, "visibility", 0.0)

        if visibility < 0.4:
            continue

        points[idx] = (x, y)

        cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)

        if show_labels:
            cv2.putText(
                frame,
                name,
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_COMPLEX,
                0.45,
                (0, 255, 255),
                1
            )

    for a, b in SKELETON_CONNECTIONS:
        if a in points and b in points:
            cv2.line(frame, points[a], points[b], (255, 0, 0), 2)

    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--no_labels", action="store_true")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Visualizing selected upper-body pose landmarks.")
    print("Press q to quit.")

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame.")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_result = pose.process(rgb)

            frame = draw_selected_skeleton(
                frame,
                pose_result.pose_landmarks,
                show_labels=not args.no_labels
            )

            cv2.putText(
                frame,
                "Landmarks used by capture/train/run scripts",
                (20, 35),
                cv2.FONT_HERSHEY_COMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.imshow("Selected Pose Landmark Visualization", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
