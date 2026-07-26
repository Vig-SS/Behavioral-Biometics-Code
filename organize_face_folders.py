#!/usr/bin/env python3
# organize_face_folders.py
# Siddhi Satheesh — CSCI 693
# Project: Behavioral Biometrics for Distinguishing Visually Similar Users
#
# Builds enroll/ and probe/ for the face baseline from a folder of videos.
#
# ONE labeling rule, matching extract_pose_from_video.py --label_from
# filename_prefix: the user label is everything BEFORE the first underscore.
# This works for BOTH naming schemes used in the project:
#     capture:  sid_day1_video_1783448027.mp4   -> user "sid"
#     swapped:  sid_swapWill_day2.mp4            -> user "sid" (the real mover)
# so the face side and the behavior side always agree on who is who.
#
# Per user: earliest-sorted clip -> enroll, the rest -> probe. A user with a
# single clip is copied into probe as well so the baseline can still run.
#
# Usage:
#   python organize_face_folders.py --videos_dir videos \
#       --enroll_dir enroll --probe_dir probe
#   python organize_face_folders.py --videos_dir swapped \
#       --enroll_dir enroll_swapped --probe_dir probe_swapped

import os
import shutil
import argparse
from collections import defaultdict

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")


def label_of(filename):
    """User label = text before the first underscore (real mover)."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    return stem.split("_")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_dir", default="videos")
    ap.add_argument("--enroll_dir", default="enroll")
    ap.add_argument("--probe_dir", default="probe")
    ap.add_argument("--clear", action="store_true",
                    help="Wipe enroll/probe dirs first (avoids stale files)")
    args = ap.parse_args()

    if not os.path.isdir(args.videos_dir):
        raise SystemExit(f"ERROR: videos dir '{args.videos_dir}' does not exist.")

    files = [f for f in sorted(os.listdir(args.videos_dir))
             if f.lower().endswith(VIDEO_EXTS)]
    if not files:
        raise SystemExit(f"ERROR: no videos found in '{args.videos_dir}'.")

    if args.clear:
        for d in (args.enroll_dir, args.probe_dir):
            if os.path.isdir(d):
                shutil.rmtree(d)

    by_user = defaultdict(list)
    for f in files:
        by_user[label_of(f)].append(f)

    os.makedirs(args.enroll_dir, exist_ok=True)
    os.makedirs(args.probe_dir, exist_ok=True)

    total_e = total_p = 0
    for user, vids in sorted(by_user.items()):
        vids.sort()
        for i, v in enumerate(vids):
            dest_root = args.enroll_dir if i == 0 else args.probe_dir
            dd = os.path.join(dest_root, user)
            os.makedirs(dd, exist_ok=True)
            shutil.copy2(os.path.join(args.videos_dir, v),
                         os.path.join(dd, v))
            if i == 0:
                total_e += 1
            else:
                total_p += 1
        if len(vids) == 1:  # single-clip user: also seed probe
            dd = os.path.join(args.probe_dir, user)
            os.makedirs(dd, exist_ok=True)
            shutil.copy2(os.path.join(args.videos_dir, vids[0]),
                         os.path.join(dd, vids[0]))
            total_p += 1
        print(f"  {user}: {len(vids)} clip(s) -> "
              f"1 enroll, {max(len(vids) - 1, 1)} probe")

    print(f"\nBuilt '{args.enroll_dir}/' ({total_e}) and "
          f"'{args.probe_dir}/' ({total_p}) for users: "
          f"{sorted(by_user.keys())}")


if __name__ == "__main__":
    main()
