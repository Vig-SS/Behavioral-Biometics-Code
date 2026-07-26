#!/usr/bin/env python3
# organize_face_swapped.py
# Siddhi Satheesh — CSCI 693
# Project: Behavioral Biometrics for Distinguishing Visually Similar Users
#
# Builds the face enroll/probe folders for the SIMULATED-TWIN (swapped)
# experiment with the scientifically correct asymmetry:
#
#   ENROLL  <- REAL videos      (the genuine face on file for each account)
#   PROBE   <- SWAPPED videos   (an impostor: real mover wearing the OTHER
#                                person's face), labeled by the REAL MOVER.
#
# Why: the paper's claim is "a look-alike/impostor fools face recognition but
# not body movement." That requires comparing a swapped probe against the
# genuine enrolled template. If BOTH enroll and probe are swapped, the faces are
# self-consistent and the matcher trivially scores 1.0 -- which tests nothing.
#
# Labeling rule (same everywhere): user = text before the first underscore.
#   real:    sid_day1_video_1783448027.mp4  -> sid   (sid's true face)
#   swapped: sid_swapWill_day2.mp4          -> sid   (sid moving, will's face)
#
# So when probe 'sid' (really sid, wearing will's face) is scored against the
# enrolled templates, matching it to WILL is a false accept -- exactly the
# failure we want to measure.
#
# Usage:
#   python organize_face_swapped.py --real_dir videos --swapped_dir swapped \
#       --enroll_dir enroll_swapped --probe_dir probe_swapped
#
#   # optional: cap how many real clips per user go into enrollment
#   python organize_face_swapped.py ... --enroll_per_user 1

import os
import shutil
import argparse
from collections import defaultdict

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")


def label_of(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    return stem.split("_")[0]


def list_videos(d):
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if f.lower().endswith(VIDEO_EXTS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", default="videos",
                    help="Folder of REAL recordings (genuine faces) for enroll")
    ap.add_argument("--swapped_dir", default="swapped",
                    help="Folder of SWAPPED clips (impostors) for probe")
    ap.add_argument("--enroll_dir", default="enroll_swapped")
    ap.add_argument("--probe_dir", default="probe_swapped")
    ap.add_argument("--enroll_per_user", type=int, default=1,
                    help="How many real clips per user to enroll (default 1)")
    ap.add_argument("--clear", action="store_true", default=True,
                    help="Wipe enroll/probe dirs first (on by default)")
    args = ap.parse_args()

    real = list_videos(args.real_dir)
    swapped = list_videos(args.swapped_dir)
    if not real:
        raise SystemExit(f"ERROR: no real videos in '{args.real_dir}'.")
    if not swapped:
        raise SystemExit(f"ERROR: no swapped videos in '{args.swapped_dir}'.")

    if args.clear:
        for d in (args.enroll_dir, args.probe_dir):
            if os.path.isdir(d):
                shutil.rmtree(d)
    os.makedirs(args.enroll_dir, exist_ok=True)
    os.makedirs(args.probe_dir, exist_ok=True)

    # ---- ENROLL from REAL videos (genuine face per user) ----
    real_by_user = defaultdict(list)
    for f in real:
        real_by_user[label_of(f)].append(f)
    print("Enrollment (REAL faces):")
    for user, vids in sorted(real_by_user.items()):
        vids.sort()
        chosen = vids[:max(1, args.enroll_per_user)]
        dd = os.path.join(args.enroll_dir, user)
        os.makedirs(dd, exist_ok=True)
        for v in chosen:
            shutil.copy2(os.path.join(args.real_dir, v), os.path.join(dd, v))
        print(f"  {user}: enrolled {len(chosen)} real clip(s) -> {chosen}")

    # ---- PROBE from SWAPPED videos (impostors), labeled by real mover ----
    swap_by_user = defaultdict(list)
    for f in swapped:
        swap_by_user[label_of(f)].append(f)
    print("Probes (SWAPPED impostors, labeled by real mover):")
    for user, vids in sorted(swap_by_user.items()):
        vids.sort()
        dd = os.path.join(args.probe_dir, user)
        os.makedirs(dd, exist_ok=True)
        for v in vids:
            shutil.copy2(os.path.join(args.swapped_dir, v), os.path.join(dd, v))
        print(f"  {user}: {len(vids)} swapped probe clip(s)")

    enrolled_users = set(real_by_user)
    probed_users = set(swap_by_user)
    if enrolled_users != probed_users:
        print(f"\nWARNING: enrolled users {sorted(enrolled_users)} != probed "
              f"users {sorted(probed_users)}. For the impostor test you want "
              f"the SAME user set on both sides so a swapped probe can be "
              f"falsely matched to the other enrolled identity.")

    print(f"\nBuilt '{args.enroll_dir}/' (real) and '{args.probe_dir}/' "
          f"(swapped). Users: {sorted(enrolled_users | probed_users)}")


if __name__ == "__main__":
    main()
