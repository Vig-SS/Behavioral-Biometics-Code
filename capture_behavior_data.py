# capture_behavior_data.py  (paper version, with video saving)
# Siddhi Satheesh — CSCI 693
# Project: Behavioral Biometrics for Distinguishing Visually Similar Users
#
# Live webcam capture. Writes the pose CSV exactly as before AND (by default)
# saves the raw video, so the same sitting can be used by BOTH the behavior
# pipeline (CSV) and the facial baseline / face-swap steps (video).
#
# The CSV and video share the SAME recording_id / filename stem, so they stay
# linked. The saved video is the CLEAN frame (no on-screen text overlay), which
# is what the face baseline and face-swap tools need.
#
# Usage:
#   python capture_behavior_data.py --user sid --session day1 --seconds 600
#   python capture_behavior_data.py --user sid --session day1 --no_video   # CSV only
#
# Outputs (default):
#   behavior_data/sid_day1_upperbody_pose_<ts>.csv
#   videos/sid_day1_video_<ts>.mp4

import os
import csv
import time
import argparse

import cv2
import mediapipe as mp

from pose_common import (
    csv_header, normalize_selected_pose_landmarks, make_pose_estimator,
)

mp_pose = mp.solutions.pose


def open_video_writer(path, width, height, fps):
    """Try mp4 first; fall back to AVI/MJPG if the mp4 codec is unavailable."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if writer.isOpened():
        return writer, path
    alt = os.path.splitext(path)[0] + ".avi"
    writer = cv2.VideoWriter(alt, cv2.VideoWriter_fourcc(*"MJPG"),
                             fps, (width, height))
    return writer, alt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--user", required=True, help="User label, e.g. sid")
    p.add_argument("--seconds", type=int, default=60)
    p.add_argument("--session", default=None,
                   help="Session tag (e.g. day1). Enables cross-session eval.")
    p.add_argument("--output_dir", default="behavior_data",
                   help="Where the pose CSV goes")
    p.add_argument("--video_dir", default="videos",
                   help="Where the raw video goes")
    p.add_argument("--no_video", action="store_true",
                   help="Do not save video (CSV only, original behavior)")
    p.add_argument("--camera", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    session = args.session or str(int(time.time()))
    stem = f"{args.user}_{session}"
    ts = int(time.time())

    csv_file = os.path.join(args.output_dir,
                            f"{stem}_upperbody_pose_{ts}.csv")
    recording_id = f"{args.user}__{session}"

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Error: Could not open webcam."); return

    cam_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = cam_fps if cam_fps and cam_fps > 1 else 20.0

    writer = None
    video_path = None
    if not args.no_video:
        os.makedirs(args.video_dir, exist_ok=True)
        video_path = os.path.join(args.video_dir, f"{stem}_video_{ts}.mp4")

    print(f"Recording '{args.user}' session '{session}' for {args.seconds}s...")
    if not args.no_video:
        print(f"Video will be saved (linked recording_id: {recording_id}).")

    start = time.time()
    with open(csv_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(csv_header())
        with make_pose_estimator(mp_pose) as pose:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame."); break

                elapsed = time.time() - start
                if elapsed >= args.seconds:
                    break

                if writer is None and not args.no_video:
                    h, wdt = frame.shape[:2]
                    writer, video_path = open_video_writer(
                        video_path, wdt, h, fps)
                    if not writer.isOpened():
                        print("Warning: could not open video writer; "
                              "continuing with CSV only.")
                        writer = None
                        args.no_video = True

                if writer is not None:
                    writer.write(frame)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)
                pose_data = normalize_selected_pose_landmarks(result.pose_landmarks)
                w.writerow([elapsed, args.user, recording_id] + pose_data)

                cv2.putText(frame, f"{args.user}/{session}: "
                            f"{max(0, args.seconds - elapsed):.1f}s left",
                            (20, 40), cv2.FONT_HERSHEY_COMPLEX, 0.75,
                            (0, 255, 0), 2)
                cv2.imshow("Capture Upper-Body Pose Data", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    print("Saved CSV:  ", csv_file)
    if video_path and os.path.exists(video_path):
        print("Saved video:", video_path)


if __name__ == "__main__":
    main()
