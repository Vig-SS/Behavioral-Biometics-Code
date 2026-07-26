# Behavioral Biometrics for Distinguishing Visually Similar Users

**Siddhi Satheesh — CSCI 693**

Code for the paper *Behavioral Biometrics for Distinguishing Visually Similar
Users*. The pipeline extracts normalized upper-body pose from ordinary webcam
video, trains a behavioral classifier, compares it against a facial-recognition
baseline, and fuses the two — evaluated both on visually distinct users and on
face-swapped "simulated twin" clips where appearance is neutralized.

---

## 0. Two virtual environments (required)

The face libraries (InsightFace/onnxruntime) and the pose library (MediaPipe)
need **conflicting versions of protobuf**, so the project uses two venvs. Set
both up once:

```bash
# pose / behavior / analysis environment
python3 -m venv .venv-pose
source .venv-pose/bin/activate
pip install -r requirements-pose.txt
deactivate

# face environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-face.txt
deactivate
```

Rule of thumb: anything touching MediaPipe uses `.venv-pose`; anything touching
faces uses `.venv`. `run_experiments.py` calls the right one per step
automatically.

One-time downloads for the face-swap step:
- The InsightFace swap model `inswapper_128.onnx` → place in
  `~/.insightface/models/` (see the header of `face_swap.py`).

---

## 1. Record data (or bring your own video)

Record several sessions per person. Each session writes a pose CSV **and** a
video (used later by the face baseline and the swap step).

```bash
source .venv-pose/bin/activate
python capture_behavior_data.py --user sid  --session day1 --seconds 600
python capture_behavior_data.py --user sid  --session day2 --seconds 600
# ...repeat per person and per day (aim for >=3 sessions each)...
```

Outputs: `behavior_data/<user>_<session>_upperbody_pose_*.csv` and
`videos/<user>_<session>_video_*.mp4`.

To use existing video files instead of live capture, drop them in `videos/`
named `user_session_video_*.mp4` and run `extract_pose_from_video.py` (see its
header for options).

---

## 2. Run the matched-condition pipeline (Experiments A + B + D)

One command runs pose extraction → behavior training → behavior evaluation →
face baseline → face evaluation → fusion, using the correct venv per step:

```bash
python run_experiments.py --mode real
#   add --device cpu   to force CPU (GPU is used automatically if available)
#   add --dry_run       to preview the plan without running
#   add --force         to re-run steps whose outputs already exist
```

Key outputs in `results/`:
- `behavior_summary.txt`, `face_summary.txt` — EER / AUC / accuracy per modality
- `behavior_confusion.csv`, `face_confusion.csv` — closed-set confusion matrices
- `*_pair_far.csv` — per-pair false-acceptance (which users get confused)
- `fusion_summary.csv` — face vs behavior vs fused
- `fusion_rescued_pairs.csv` — impostors face accepts but fusion rejects

---

## 3. Generate the simulated twins (Experiment C, face venv)

Face-swap makes visually-similar pairs without recruiting twins: each person's
body wears another person's face, so they *look* alike but *move* differently.
The swap only touches the face region, so the pose signal stays genuine.

```bash
source .venv/bin/activate
mkdir -p faces swapped

# one clear face image per person (any front-facing frame works)
ffmpeg -y -i videos/sid_day1_video_*.mp4  -vf "select=eq(n\,150)" -frames:v 1 -update 1 faces/sid.jpg
ffmpeg -y -i videos/will_day1_video_*.mp4 -vf "select=eq(n\,150)" -frames:v 1 -update 1 faces/will.jpg
ffmpeg -y -i videos/chel_day1_video_*.mp4 -vf "select=eq(n\,150)" -frames:v 1 -update 1 faces/chel.jpg

# 3-cycle swaps for day2..day6 (day1 stays real, for enrollment)
for d in day2 day3 day4 day5 day6; do
  python face_swap.py --source_image faces/will.jpg --target_video videos/sid_${d}_video_*.mp4  --out swapped/sid_swapWill_${d}.mp4  --device gpu
  python face_swap.py --source_image faces/chel.jpg --target_video videos/will_${d}_video_*.mp4 --out swapped/will_swapChel_${d}.mp4 --device gpu
  python face_swap.py --source_image faces/sid.jpg  --target_video videos/chel_${d}_video_*.mp4 --out swapped/chel_swapSid_${d}.mp4 --device gpu
done
```

Naming rule: the prefix before the first `_` must be the **real mover**
(`sid_swapWill_*` → sid), because both the pose labeling and the face
organization use it.

Verify a swap actually fools the matcher before trusting it (high similarity to
the worn face, low to the mover's own):

```bash
python make_simulated_twins.py --verify --swapped swapped/sid_swapWill_day2.mp4 \
    --target_ref faces/will.jpg --other_ref faces/sid.jpg
```

---

## 4. Run the swapped-condition pipeline (Experiment C eval + D)

```bash
python run_experiments.py --mode swapped
```

This extracts pose from the swapped clips, trains/evaluates behavior on them,
builds **real-enroll / swapped-probe** face folders
(`organize_face_swapped.py`), runs the face baseline, and fuses. Outputs land in
`results_swapped/`. The expected result: face collapses (high EER, misidentifies
the swapped probes) while behavior is unaffected.

Optional fusion weight sweep:

```bash
for w in 0.5 0.4 0.3 0.2 0.1; do
  python fuse_scores.py --face face_scores_swapped.csv \
      --behavior behavior_comparisons_swapped.csv \
      --w_face $w --w_behavior $(echo "1 - $w" | bc) \
      --target_far 0.05 --outdir results_swapped_w
done
```

---

## File reference

| File | Env | Role |
|---|---|---|
| `pose_common.py` | pose | Shared landmarks, normalization, window features (single source of truth) |
| `capture_behavior_data.py` | pose | Live webcam capture → pose CSV + saved video |
| `extract_pose_from_video.py` | pose | Pose CSVs from video files (DAiSEE / YouTube / swapped) |
| `train_behavior_classifier.py` | pose | Train behavior classifier; export per-window scores |
| `run_behavior_classifier.py` | pose | Live demo of a trained classifier |
| `behavior_to_comparisons.py` | pose | Behavior probabilities → genuine/impostor comparisons |
| `verification_metrics.py` | pose | FAR/FRR, EER, ROC AUC helpers |
| `evaluate.py` | pose | Per-modality metrics + confusion + per-pair FAR |
| `fuse_scores.py` | pose | Score-level fusion of face + behavior |
| `visualize_pose_points.py` | pose | Debug overlay of tracked landmarks |
| `face_baseline.py` | face | Facial-ID baseline (GPU-aware) |
| `face_swap.py` | face | Region-limited face swap (inswapper) |
| `make_simulated_twins.py` | face | Swap recipe + verification check |
| `organize_face_folders.py` | either | Build enroll/probe from a video folder |
| `organize_face_swapped.py` | either | Build real-enroll / swapped-probe folders |
| `run_experiments.py` | both | Orchestrates the whole pipeline, correct venv per step |

---

## Keeping requirements current

To capture the exact versions you actually ran (for reproducibility), freeze
each environment:

```bash
source .venv-pose/bin/activate && pip freeze > requirements-pose.lock.txt
source .venv/bin/activate      && pip freeze > requirements-face.lock.txt
```

The `requirements-*.txt` files list the minimal direct dependencies; the
`.lock.txt` files (if you generate them) pin every transitive version.
