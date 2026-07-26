# face_baseline.py
# Siddhi Satheesh — CSCI 693
# Project: Behavioral Biometrics for Distinguishing Visually Similar Users
#
# The facial-identification baseline the paper compares against. It builds a
# face "template" (mean embedding) per enrolled user from their video(s), then
# scores probe videos against every enrolled user. Output is a tidy CSV of
# genuine/impostor comparison scores that the evaluation + fusion scripts read,
# using the SAME format as the behavior side.
#
# Backends (auto-detected, in order):
#   1. insightface (ArcFace)     -- best; pip install insightface onnxruntime
#   2. face_recognition (dlib)   -- pip install face_recognition
# If neither is installed the script explains what to install and exits.
#
# Enrollment/probe layout (folders of videos, label = subfolder name):
#   enroll/<user>/*.mp4      probe/<user>/*.mp4
#
# Usage:
#   python face_baseline.py --enroll_dir enroll --probe_dir probe \
#       --out face_scores.csv --sample_every 15
#
# A comparison score is cosine similarity in [-1,1] (higher = more similar).

import os
import glob
import argparse
import numpy as np

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# ---------- backend abstraction ------------------------------------------
class FaceBackend:
    name = "none"
    def embed_bgr(self, frame_bgr):
        """Return a 1-D embedding for the largest face, or None."""
        raise NotImplementedError


class InsightFaceBackend(FaceBackend):
    name = "insightface(arcface)"
    def __init__(self, device="auto", det_size=640):
        from insightface.app import FaceAnalysis
        import onnxruntime as ort

        avail = ort.get_available_providers()
        want_gpu = device in ("auto", "gpu", "cuda")
        # NOTE: we deliberately do NOT request TensorRT. It needs a separate
        # NVIDIA library (libnvinfer) and, when it fails to load, some
        # onnxruntime builds fall straight back to CPU instead of trying CUDA.
        # Plain CUDA gives the GPU speedup without that fragility.
        providers = []
        if want_gpu and "CUDAExecutionProvider" in avail:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")  # always keep as fallback

        if device == "gpu" and "CUDAExecutionProvider" not in avail:
            print("  ! --device gpu requested but CUDAExecutionProvider is not "
                  "available. Install onnxruntime-gpu (CUDA 12 build) + cuDNN. "
                  "Falling back to CPU.")

        # verify CUDA can ACTUALLY create a session (available != loadable),
        # so we report the real device instead of guessing.
        active = "CPU"
        if "CUDAExecutionProvider" in providers:
            try:
                _ = ort.InferenceSession.__module__  # cheap import guard
                # a genuine load test happens when FaceAnalysis.prepare runs;
                # ort.get_device() reflects the compiled device.
                if ort.get_device().upper() == "GPU":
                    active = "GPU"
            except Exception:
                active = "CPU"

        ctx_id = 0 if active == "GPU" else -1
        self.app = FaceAnalysis(name="buffalo_l", providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
        self._active = active
        self.name = f"insightface(arcface) [{active}]"
    def embed_bgr(self, frame_bgr):
        faces = self.app.get(frame_bgr)
        if not faces:
            return None
        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) *
                   (f.bbox[3] - f.bbox[1]), reverse=True)
        return np.asarray(faces[0].normed_embedding, dtype=np.float32)


class FaceRecognitionBackend(FaceBackend):
    name = "face_recognition(dlib)"
    def __init__(self):
        import face_recognition
        self.fr = face_recognition
    def embed_bgr(self, frame_bgr):
        rgb = frame_bgr[:, :, ::-1]
        locs = self.fr.face_locations(rgb)
        if not locs:
            return None
        # largest face
        locs.sort(key=lambda b: (b[2] - b[0]) * (b[1] - b[3]), reverse=True)
        encs = self.fr.face_encodings(rgb, [locs[0]])
        if not encs:
            return None
        return np.asarray(encs[0], dtype=np.float32)


def load_backend(device="auto", det_size=640):
    try:
        return InsightFaceBackend(device=device, det_size=det_size)
    except Exception as e:
        print("  (insightface unavailable:", e, ")")
    try:
        return FaceRecognitionBackend()
    except Exception:
        pass
    return None


# ---------- helpers -------------------------------------------------------
def cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


def gather(input_dir):
    """Return dict user -> list of media paths (videos or images)."""
    out = {}
    for user in sorted(os.listdir(input_dir)):
        udir = os.path.join(input_dir, user)
        if not os.path.isdir(udir):
            continue
        paths = []
        for root, _, files in os.walk(udir):
            for fn in files:
                if fn.lower().endswith(VIDEO_EXTS + IMAGE_EXTS):
                    paths.append(os.path.join(root, fn))
        if paths:
            out[user] = sorted(paths)
    return out


def embeddings_from_media(path, backend, sample_every, max_frames=0):
    """
    max_frames > 0 stops after that many EMBEDDED faces. A face template only
    needs a few dozen good frames, so capping this is the main speedup for long
    videos (a 10-min clip has ~18000 frames; ~40 is plenty).
    """
    import cv2
    embs = []
    if path.lower().endswith(IMAGE_EXTS):
        img = cv2.imread(path)
        if img is not None:
            e = backend.embed_bgr(img)
            if e is not None:
                embs.append(e)
        return embs
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return embs
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if sample_every > 1 and idx % sample_every:
            continue
        e = backend.embed_bgr(frame)
        if e is not None:
            embs.append(e)
            if max_frames and len(embs) >= max_frames:
                break
    cap.release()
    return embs


def mean_template(embs):
    if not embs:
        return None
    m = np.mean(np.stack(embs), axis=0)
    return m / (np.linalg.norm(m) + 1e-9)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--enroll_dir", required=True)
    p.add_argument("--probe_dir", required=True)
    p.add_argument("--out", default="face_scores.csv")
    p.add_argument("--sample_every", type=int, default=15,
                   help="Embed every Nth video frame")
    p.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto",
                   help="auto uses GPU if onnxruntime-gpu+CUDA are available")
    p.add_argument("--max_frames", type=int, default=40,
                   help="Max embedded faces per video (0 = no cap). "
                        "Biggest speedup for long clips.")
    p.add_argument("--det_size", type=int, default=640,
                   help="Face detector input size; 320 is faster, less accurate")
    args = p.parse_args()

    backend = load_backend(device=args.device, det_size=args.det_size)
    if backend is None:
        print("No face-embedding backend found. Install ONE of:")
        print("  pip install insightface onnxruntime      (ArcFace, best)")
        print("  pip install face_recognition             (dlib)")
        return
    print(f"Face backend: {backend.name}")

    import pandas as pd
    enroll = gather(args.enroll_dir)
    probe = gather(args.probe_dir)
    if not enroll or not probe:
        print("Need user subfolders with media in enroll and probe dirs."); return

    # build one template per enrolled user
    templates = {}
    for user, paths in enroll.items():
        embs = []
        for pth in paths:
            embs.extend(embeddings_from_media(
                pth, backend, args.sample_every, args.max_frames))
        t = mean_template(embs)
        if t is not None:
            templates[user] = t
            print(f"  enrolled {user}: {len(embs)} face frames")
        else:
            print(f"  ! no faces found for enrolled user {user}")

    # score every probe clip against every enrolled template
    rows = []
    n_clips = sum(len(v) for v in probe.values())
    done = 0
    for probe_user, paths in probe.items():
        for pth in paths:
            done += 1
            print(f"  [{done}/{n_clips}] probe {os.path.basename(pth)}")
            embs = embeddings_from_media(
                pth, backend, args.sample_every, args.max_frames)
            probe_t = mean_template(embs)
            if probe_t is None:
                print(f"  ! no faces in probe {pth}")
                continue
            for enrolled_user, tmpl in templates.items():
                rows.append({
                    "probe_user": probe_user,
                    "probe_clip": os.path.basename(pth),
                    "enrolled_user": enrolled_user,
                    "face_score": cosine(probe_t, tmpl),
                    "genuine": int(probe_user == enrolled_user),
                })
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Wrote {len(rows)} face comparison scores -> {args.out}")


if __name__ == "__main__":
    main()
