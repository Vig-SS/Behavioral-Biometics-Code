#!/usr/bin/env python3
# face_swap.py
# Siddhi Satheesh — CSCI 693
# Project: Behavioral Biometrics for Distinguishing Visually Similar Users
#
# Region-limited face swap for the "simulated twins" experiment, built on the
# SAME InsightFace + onnxruntime stack already used by face_baseline.py (so it
# runs on the GPU you already set up, in the .venv face environment).
#
# It replaces ONLY the face region of every frame in --target_video with the
# face from --source_image, leaving the body/torso motion untouched. That is
# what keeps the behavioral (pose) signal authentic while neutralizing
# appearance -- the core requirement of the simulated-twin design.
#
# Run this in the FACE venv (the one with insightface + onnxruntime-gpu):
#
#   # will's face onto sid's body videos  -> looks like will, moves like sid
#   python face_swap.py --source_image faces/will.jpg \
#       --target_video videos/sid_day2_video_XXXX.mp4 \
#       --out swapped/sid_swapWill_day2.mp4 --device gpu
#
#   # sid's face onto will's body videos  -> looks like sid, moves like will
#   python face_swap.py --source_image faces/sid.jpg \
#       --target_video videos/will_day2_video_YYYY.mp4 \
#       --out swapped/will_swapSid_day2.mp4 --device gpu
#
# IMPORTANT naming rule for the pose step:
#   name outputs <REALMOVER>_swap...  so --label_from filename_prefix labels the
#   clip by who actually MOVED (sid_swapWill -> label 'sid'). The pose classifier
#   must learn the real mover, not the face that was painted on.

import os
import cv2
import argparse
import numpy as np


def build_app_and_swapper(device):
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import get_model
    import onnxruntime as ort

    avail = ort.get_available_providers()
    providers = []
    if device in ("auto", "gpu") and "CUDAExecutionProvider" in avail:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    ctx = 0 if (providers[0] == "CUDAExecutionProvider") else -1

    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=ctx, det_size=(640, 640))

    # inswapper_128.onnx is the standard InsightFace swap model.
    model_path = os.path.expanduser(
        "~/.insightface/models/inswapper_128.onnx")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Missing inswapper_128.onnx.\n"
            "Download it once and place it at:\n  " + model_path + "\n"
            "It is widely mirrored; search 'inswapper_128.onnx'. "
            "Put the file in ~/.insightface/models/.")
    swapper = get_model(model_path, providers=providers)
    active = "GPU" if ctx == 0 else "CPU"
    return app, swapper, active


def pick_source_face(app, image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read source image: {image_path}")
    faces = app.get(img)
    if not faces:
        raise RuntimeError(f"No face found in source image: {image_path}")
    faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]),
               reverse=True)
    return faces[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source_image", required=True,
                   help="Image of the face to paint ON (the target identity)")
    p.add_argument("--target_video", required=True,
                   help="Video whose BODY MOTION is kept; only face is replaced")
    p.add_argument("--out", required=True, help="Output swapped video path")
    p.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    p.add_argument("--every", type=int, default=1,
                   help="Swap every Nth frame (1=all). Higher = faster preview.")
    args = p.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    app, swapper, active = build_app_and_swapper(args.device)
    print(f"Swap backend: insightface inswapper [{active}]")

    source_face = pick_source_face(app, args.source_image)

    cap = cv2.VideoCapture(args.target_video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open target video: {args.target_video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))
    if not writer.isOpened():
        alt = os.path.splitext(args.out)[0] + ".avi"
        writer = cv2.VideoWriter(alt, cv2.VideoWriter_fourcc(*"MJPG"),
                                 fps, (w, h))
        print("  (mp4 codec unavailable; writing", alt, ")")
        args.out = alt

    n, swapped, no_face = 0, 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        n += 1
        out_frame = frame
        if args.every <= 1 or n % args.every == 0:
            faces = app.get(frame)
            if faces:
                faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]),
                           reverse=True)
                # paste_back=True replaces ONLY the face region in the frame
                out_frame = swapper.get(frame, faces[0], source_face,
                                        paste_back=True)
                swapped += 1
            else:
                no_face += 1
        writer.write(out_frame)
        if n % 100 == 0:
            print(f"  frame {n} (swapped {swapped}, no-face {no_face})")

    cap.release()
    writer.release()
    print(f"Done: {args.out}  ({n} frames, {swapped} swapped, "
          f"{no_face} had no detectable face)")


if __name__ == "__main__":
    main()
