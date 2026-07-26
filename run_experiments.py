#!/usr/bin/env python3
# run_experiments.py
# Siddhi Satheesh — CSCI 693
# Project: Behavioral Biometrics for Distinguishing Visually Similar Users
#
# One command to run a full experiment set end to end, with the correct venv
# per step and existence checks between steps.
#
# Face steps use .venv (insightface + onnxruntime-gpu); pose steps use
# .venv-pose (mediapipe). Each step is launched with its venv's python directly,
# checks its inputs first, verifies its outputs after, and by default skips
# steps whose outputs already exist (so re-running resumes where you left off).
#
# IMPORTANT (swapped mode): the face baseline enrolls from REAL videos and
# probes from SWAPPED clips. That asymmetry is what tests the paper's claim --
# a swapped impostor should be FALSELY matched to the genuine enrolled identity.
# (Enrolling AND probing on swapped clips trivially scores 1.0 and tests
# nothing.)
#
# Usage:
#   python run_experiments.py --mode real
#   python run_experiments.py --mode swapped
#   python run_experiments.py --mode swapped --dry_run
#   python run_experiments.py --mode swapped --force

import os
import argparse
import subprocess


def gpu_env_for_face(face_python):
    """
    Build an environment dict that lets onnxruntime-gpu find the CUDA/cuDNN
    libraries bundled inside the FACE venv. run_experiments launches the venv's
    python directly (not via `activate`), so LD_LIBRARY_PATH is not set the way
    it would be in an interactive shell -- which is why the GPU silently fell
    back to CPU. We ask the face interpreter itself where the nvidia .so files
    live and prepend those dirs to LD_LIBRARY_PATH. If nothing is found (e.g.
    CPU-only onnxruntime), we return the environment unchanged.
    """
    probe = (
        "import os, importlib\n"
        "d=[]\n"
        "for m in ['nvidia.cudnn','nvidia.cublas','nvidia.cuda_runtime',"
        "'nvidia.cuda_nvrtc','nvidia.cufft','nvidia.curand']:\n"
        "    try: mod=importlib.import_module(m)\n"
        "    except Exception: continue\n"
        "    base=os.path.dirname(mod.__file__) if getattr(mod,'__file__',None) "
        "else (list(mod.__path__)[0] if getattr(mod,'__path__',None) else None)\n"
        "    if base and os.path.isdir(os.path.join(base,'lib')): "
        "d.append(os.path.join(base,'lib'))\n"
        "print(os.pathsep.join(d))\n"
    )
    env = os.environ.copy()
    try:
        out = subprocess.run([face_python, "-c", probe],
                             capture_output=True, text=True, timeout=60)
        libdirs = out.stdout.strip()
    except Exception:
        libdirs = ""
    if libdirs:
        prev = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = libdirs + (os.pathsep + prev if prev else "")
    return env, libdirs

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")


def has_video(d):
    return os.path.isdir(d) and any(
        f.lower().endswith(VIDEO_EXTS) for f in os.listdir(d))


def has_csv(d):
    return os.path.isdir(d) and any(f.endswith(".csv") for f in os.listdir(d))


def nonempty_file(p):
    return os.path.isfile(p) and os.path.getsize(p) > 0


def has_video_tree(root):
    if not os.path.isdir(root):
        return False
    for user in os.listdir(root):
        ud = os.path.join(root, user)
        if os.path.isdir(ud) and any(
                f.lower().endswith(VIDEO_EXTS) for f in os.listdir(ud)):
            return True
    return False


class Step:
    def __init__(self, name, venv, argv, needs, makes, check_makes=None):
        self.name = name
        self.venv = venv
        self.argv = argv
        self.needs = needs
        self.makes = makes
        self.check_makes = check_makes or (lambda: _exists(self.makes))


def _exists(p):
    if p.endswith("/"):
        return os.path.isdir(p) and bool(os.listdir(p))
    return nonempty_file(p) or (os.path.isdir(p) and bool(os.listdir(p)))


def build_plan(v, mode):
    # face enroll/probe organization differs by mode
    if mode == "swapped":
        organize_step = Step(
            "B1 organize face (REAL enroll / SWAPPED probe)", "pose",
            ["organize_face_swapped.py", "--real_dir", v["real_videos"],
             "--swapped_dir", v["videos"], "--enroll_dir", v["enroll"],
             "--probe_dir", v["probe"]],
            needs=[(f"real videos in {v['real_videos']}/",
                    lambda: has_video(v["real_videos"])),
                   (f"swapped videos in {v['videos']}/",
                    lambda: has_video(v["videos"]))],
            makes=v["enroll"] + "/",
            check_makes=lambda: has_video_tree(v["enroll"]) and
            has_video_tree(v["probe"]))
    else:
        organize_step = Step(
            "B1 organize face enroll/probe", "pose",
            ["organize_face_folders.py", "--videos_dir", v["videos"],
             "--enroll_dir", v["enroll"], "--probe_dir", v["probe"],
             "--clear"],
            needs=[(f"videos in {v['videos']}/",
                    lambda: has_video(v["videos"]))],
            makes=v["enroll"] + "/",
            check_makes=lambda: has_video_tree(v["enroll"]))

    steps = [
        Step("A1 extract pose from videos", "pose",
             ["extract_pose_from_video.py", "--input_dir", v["videos"],
              "--label_from", "filename_prefix", "--output_dir", v["beh_data"]],
             needs=[(f"videos in {v['videos']}/",
                     lambda: has_video(v["videos"]))],
             makes=v["beh_data"] + "/",
             check_makes=lambda: has_csv(v["beh_data"])),

        Step("A2 train behavior classifier", "pose",
             ["train_behavior_classifier.py", "--data_dir", v["beh_data"],
              "--model_out", v["clf"], "--scores_out", v["beh_scores"],
              "--window_size", "150", "--stride", "30"],
             needs=[(f"pose CSVs in {v['beh_data']}/",
                     lambda: has_csv(v["beh_data"]))],
             makes=v["beh_scores"]),

        Step("A3 behavior -> comparisons", "pose",
             ["behavior_to_comparisons.py", "--scores", v["beh_scores"],
              "--out", v["beh_cmp"]],
             needs=[(v["beh_scores"], lambda: nonempty_file(v["beh_scores"]))],
             makes=v["beh_cmp"]),

        Step("A4 evaluate behavior", "pose",
             ["evaluate.py", "--comparisons", v["beh_cmp"],
              "--score_col", "behavior_score", "--probe_col", "probe_user",
              "--candidate_col", "candidate_user", "--tag", "behavior",
              "--target_far", str(v["target_far"]), "--outdir", v["outdir"]],
             needs=[(v["beh_cmp"], lambda: nonempty_file(v["beh_cmp"]))],
             makes=os.path.join(v["outdir"], "behavior_summary.txt")),

        organize_step,

        Step("B2 face baseline (GPU)", "face",
             ["face_baseline.py", "--enroll_dir", v["enroll"],
              "--probe_dir", v["probe"], "--out", v["face_scores"],
              "--sample_every", "30", "--max_frames", "40",
              "--device", v["device"]],
             needs=[(f"enroll tree {v['enroll']}/",
                     lambda: has_video_tree(v["enroll"])),
                    (f"probe tree {v['probe']}/",
                     lambda: has_video_tree(v["probe"]))],
             makes=v["face_scores"]),

        Step("B3 evaluate face", "pose",
             ["evaluate.py", "--comparisons", v["face_scores"],
              "--score_col", "face_score", "--probe_col", "probe_user",
              "--candidate_col", "enrolled_user", "--tag", "face",
              "--target_far", str(v["target_far"]), "--outdir", v["outdir"]],
             needs=[(v["face_scores"], lambda: nonempty_file(v["face_scores"]))],
             makes=os.path.join(v["outdir"], "face_summary.txt")),

        Step("D  fusion (face + behavior)", "pose",
             ["fuse_scores.py", "--face", v["face_scores"],
              "--behavior", v["beh_cmp"], "--w_face", "0.5",
              "--w_behavior", "0.5", "--target_far", str(v["target_far"]),
              "--outdir", v["outdir"]],
             needs=[(v["face_scores"], lambda: nonempty_file(v["face_scores"])),
                    (v["beh_cmp"], lambda: nonempty_file(v["beh_cmp"]))],
             makes=os.path.join(v["outdir"], "fusion_summary.csv")),
    ]
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["real", "swapped"], required=True)
    ap.add_argument("--face_python", default=".venv/bin/python")
    ap.add_argument("--pose_python", default=".venv-pose/bin/python")
    ap.add_argument("--device", default="gpu", choices=["auto", "gpu", "cpu"])
    ap.add_argument("--target_far", type=float, default=0.05)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    if args.mode == "real":
        cfg = dict(videos="videos", real_videos="videos",
                   beh_data="behavior_data",
                   clf="behavior_classifier.joblib",
                   beh_scores="behavior_scores.csv",
                   beh_cmp="behavior_comparisons.csv",
                   enroll="enroll", probe="probe",
                   face_scores="face_scores.csv", outdir="results")
    else:
        cfg = dict(videos="swapped", real_videos="videos",
                   beh_data="behavior_data_swapped",
                   clf="clf_swapped.joblib",
                   beh_scores="behavior_scores_swapped.csv",
                   beh_cmp="behavior_comparisons_swapped.csv",
                   enroll="enroll_swapped", probe="probe_swapped",
                   face_scores="face_scores_swapped.csv",
                   outdir="results_swapped")
    cfg["device"] = args.device
    cfg["target_far"] = args.target_far

    interp = {"face": args.face_python, "pose": args.pose_python}
    for role, path in interp.items():
        if not args.dry_run and not os.path.exists(path):
            print(f"ERROR: {role} interpreter not found: {path}\n"
                  f"       Pass --{role}_python /path/to/venv/bin/python")
            return

    os.makedirs(cfg["outdir"], exist_ok=True)
    steps = build_plan(cfg, args.mode)

    print(f"=== Pipeline: mode={args.mode}  device={args.device} ===")
    for i, s in enumerate(steps, 1):
        print(f"\n[{i}/{len(steps)}] {s.name}   (venv: {s.venv})")

        if not args.force and s.check_makes():
            print(f"    skip: output already exists ({s.makes})")
            continue

        missing = [desc for desc, ok in s.needs if not ok()]
        if missing:
            if args.dry_run:
                print("    (inputs not present yet; a prior step would create "
                      "them):")
                for m in missing:
                    print(f"      - {m}")
            else:
                print("    MISSING INPUTS:")
                for m in missing:
                    print(f"      - {m}")
                print("    Stopping. Fix the step that produces the missing "
                      "input, then re-run (finished steps are skipped).")
                return

        cmd = [interp[s.venv]] + s.argv
        print("    $", " ".join(cmd))
        if args.dry_run:
            continue

        # For the GPU-capable face step, inject the venv's CUDA/cuDNN lib paths
        # so onnxruntime-gpu can actually load CUDA (mirrors what `activate`
        # would do). Pose steps and CPU runs are unaffected.
        step_env = None
        if s.venv == "face" and args.device in ("auto", "gpu"):
            step_env, libdirs = gpu_env_for_face(interp["face"])
            if libdirs:
                print("    (CUDA libs found; enabling GPU for this step)")
            else:
                print("    (no bundled CUDA libs found; face step will use CPU)")

        rc = subprocess.run(cmd, env=step_env).returncode
        if rc != 0:
            print(f"    STEP FAILED (exit {rc}). Nothing after this ran.")
            return
        if not s.check_makes():
            print(f"    STEP RAN but expected output is missing: {s.makes}")
            return

    print("\n=== DONE ===")
    if not args.dry_run:
        print("Key results in", cfg["outdir"] + "/ :")
        for fn in ["behavior_summary.txt", "face_summary.txt",
                   "face_pair_far.csv", "fusion_summary.csv",
                   "fusion_rescued_pairs.csv"]:
            p = os.path.join(cfg["outdir"], fn)
            print(("  [ok] " if os.path.exists(p) else "  [--] ") + p)


if __name__ == "__main__":
    main()
