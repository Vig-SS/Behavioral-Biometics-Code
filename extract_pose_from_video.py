# extract_pose_from_video.py
# Siddhi Satheesh — CSCI 693
# Project: Behavioral Biometrics for Distinguishing Visually Similar Users
#
# Turns VIDEO FILES into the SAME pose CSV format your live capture produces,
# so DAiSEE clips, YouTube clips, and face-swapped videos all flow into the
# exact training/eval pipeline you already have.
#
# Key idea: the only difference from capture_behavior_data.py is the frame
# source (a file instead of a webcam index) and that we can label the "user"
# and a "session" from the folder/filename rather than a --user flag.
#
# Examples
#   Single file, explicit label:
#     python extract_pose_from_video.py --video clips/sid_a.mp4 --user sid \
#         --session a --output_dir behavior_data
#
#   A folder laid out as  root/<user>/<anything>.mp4  (label = subfolder name):
#     python extract_pose_from_video.py --input_dir daisee_clips \
#         --label_from folder --output_dir behavior_data
#
#   A flat folder where the label is the part of the filename before "_":
#     python extract_pose_from_video.py --input_dir swapped \
#         --label_from filename_prefix --output_dir behavior_data

import os
import csv
import glob
import time
import argparse

import cv2
import mediapipe as mp

from pose_common import (
    POSE_LANDMARKS_USED, csv_header, normalize_selected_pose_landmarks,
    make_pose_estimator,
)

mp_pose = mp.solutions.pose

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")


def label_for(path, mode, fixed_user):
    base = os.path.basename(path)
    if mode == "folder":
        # parent directory name is the user label
        return os.path.basename(os.path.dirname(os.path.abspath(path)))
    if mode == "filename_prefix":
        # text before the first underscore, e.g. sid_a.mp4 -> sid
        return os.path.splitext(base)[0].split("_")[0]
    return fixed_user  # "fixed"


def session_for(path, explicit):
    if explicit is not None:
        return explicit
    # default session id = filename without extension (keeps recordings distinct)
    return os.path.splitext(os.path.basename(path))[0]


def process_one(video_path, user, session, output_dir, sample_every,
                model_complexity, show):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ! could not open {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    os.makedirs(output_dir, exist_ok=True)
    stamp = int(time.time() * 1000)
    out_file = os.path.join(
        output_dir, f"{user}_{session}_video_pose_{stamp}.csv"
    )
    recording_id = f"{user}__{session}"   # stable group id for eval splitting

    header = csv_header()
    written = 0
    frame_idx = 0

    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        with make_pose_estimator(mp_pose, model_complexity) as pose:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                if sample_every > 1 and (frame_idx % sample_every) != 0:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)
                pose_data = normalize_selected_pose_landmarks(result.pose_landmarks)

                timestamp = frame_idx / fps
                writer.writerow([timestamp, user, recording_id] + pose_data)
                written += 1

                if show:
                    cv2.putText(frame, f"{user}/{session}", (20, 40),
                                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.imshow("extract", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    cap.release()
    if show:
        cv2.destroyAllWindows()

    print(f"  -> {out_file}  ({written} frames, user={user}, session={session})")
    return out_file


def gather_videos(input_dir):
    vids = []
    for root, _, files in os.walk(input_dir):
        for fn in files:
            if fn.lower().endswith(VIDEO_EXTS):
                vids.append(os.path.join(root, fn))
    return sorted(vids)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", help="Path to a single video file")
    p.add_argument("--input_dir", help="Folder of videos (searched recursively)")
    p.add_argument("--output_dir", default="behavior_data")
    p.add_argument("--user", default="user",
                   help="User label when --label_from fixed (single video)")
    p.add_argument("--session", default=None,
                   help="Session id; default = filename without extension")
    p.add_argument("--label_from", choices=["fixed", "folder", "filename_prefix"],
                   default="fixed",
                   help="How to derive the user label for --input_dir batches")
    p.add_argument("--sample_every", type=int, default=1,
                   help="Process every Nth frame (1 = all frames)")
    p.add_argument("--model_complexity", type=int, default=1, choices=[0, 1, 2])
    p.add_argument("--show", action="store_true", help="Preview window")
    args = p.parse_args()

    if not args.video and not args.input_dir:
        p.error("provide --video or --input_dir")

    targets = []
    if args.video:
        targets.append(args.video)
    if args.input_dir:
        targets.extend(gather_videos(args.input_dir))

    if not targets:
        print("No videos found.")
        return

    print(f"Processing {len(targets)} video(s)...")
    for path in targets:
        user = label_for(path, args.label_from, args.user)
        session = session_for(path, args.session if args.video else None)
        process_one(path, user, session, args.output_dir,
                    args.sample_every, args.model_complexity, args.show)

    print("Done. CSVs written to:", args.output_dir)


if __name__ == "__main__":
    main()
