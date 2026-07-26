# make_simulated_twins.py
# Siddhi Satheesh — CSCI 693
# Project: Behavioral Biometrics for Distinguishing Visually Similar Users
#
# "Simulated twins" for the paper: take real videos from DIFFERENT movers and
# give them the SAME face, so appearance is neutralized while each subject's
# genuine BODY motion is preserved. Because the swap only rewrites the face
# region, the upper-body pose signal your pipeline uses stays authentic.
#
# This script does two things:
#   (1) --verify : confirm a swap worked as intended for the experiment, i.e.
#       the face matcher now treats the swapped clip as the TARGET identity
#       (appearance fooled) -- a required check before trusting any result.
#   (2) prints the recommended generation commands, because the actual swap is
#       best done with a dedicated, region-limited tool (facefusion / roop /
#       SimSwap / InsightFace inswapper). Full-head REENACTMENT tools are avoided
#       on purpose: they regenerate the torso and would corrupt the motion
#       signal you are trying to measure.
#
# Recommended external tool (region-limited face swap, keeps body untouched):
#   pip install facefusion            # or use InsightFace 'inswapper_128.onnx'
#   facefusion headless-run \
#       --source target_face.jpg \
#       --target real_mover_clip.mp4 \
#       --output swapped_moverA_as_target.mp4 \
#       --face-swapper-model inswapper_128 --processors face_swapper
#
# Verify the swap fooled the matcher (uses face_baseline's backend):
#   python make_simulated_twins.py --verify \
#       --swapped swapped_moverA_as_target.mp4 \
#       --target_ref target_face.jpg --sample_every 15

import argparse
import numpy as np

from face_baseline import load_backend, embeddings_from_media, mean_template, cosine


GUIDE = """
Simulated-twin recipe (safe design for the paper)
-------------------------------------------------
Goal: two DIFFERENT real movers that LOOK identical.

1. Record/collect real clips of movers A and B (their genuine motion):
     realA.mp4   (A's body motion)
     realB.mp4   (B's body motion)

2. Pick one shared target face image: target_face.jpg
     (could be a synthetic face, or A's own face).

3. Region-limited face swap (body untouched) with facefusion/inswapper:
     swapped_A_as_T.mp4   = target face on A's body/motion
     swapped_B_as_T.mp4   = target face on B's body/motion
   Now both LOOK like T; A and B still MOVE as themselves.

4. VERIFY appearance is neutralized (this script, --verify):
     face matcher should score swapped_A_as_T and swapped_B_as_T as the SAME
     identity (high similarity to target). If it does NOT, the swap is too weak
     and the pair is not a valid simulated twin -- redo it.

5. Extract pose (extract_pose_from_video.py) from the SWAPPED clips, labeling
     user=A for swapped_A_as_T and user=B for swapped_B_as_T. Behavior should
     still separate A vs B even though the face baseline cannot.

Impostor variant ("someone appearing as me"):
   Put YOUR face on someone else's body clip -> face matcher accepts (looks like
   you), behavior should REJECT (moves like them). Great fusion demonstration.

Caveat to keep valid: only swap the FACE region. Do NOT use full-head/full-body
reenactment; that regenerates torso motion and contaminates the behavior signal.
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true")
    p.add_argument("--swapped", help="Swapped clip to check")
    p.add_argument("--target_ref", help="Reference image/clip of target identity")
    p.add_argument("--other_ref", help="Optional: a different identity to contrast")
    p.add_argument("--sample_every", type=int, default=15)
    p.add_argument("--accept_threshold", type=float, default=0.4,
                   help="Cosine sim above which faces are 'same identity'")
    args = p.parse_args()

    if not args.verify:
        print(GUIDE)
        return

    if not args.swapped or not args.target_ref:
        print("--verify needs --swapped and --target_ref"); return
    backend = load_backend()
    if backend is None:
        print("Install a face backend first (see face_baseline.py)."); return
    print(f"Face backend: {backend.name}")

    sw = mean_template(embeddings_from_media(args.swapped, backend,
                                             args.sample_every))
    tg = mean_template(embeddings_from_media(args.target_ref, backend,
                                             args.sample_every))
    if sw is None or tg is None:
        print("Could not get face embeddings from one of the inputs."); return

    sim_target = cosine(sw, tg)
    print(f"cosine(swapped, target) = {sim_target:.3f}")
    verdict = ("PASS: swap looks like the target -> appearance neutralized"
               if sim_target >= args.accept_threshold else
               "FAIL: swap does not match target -> not a valid simulated twin")
    print(verdict)

    if args.other_ref:
        ot = mean_template(embeddings_from_media(args.other_ref, backend,
                                                 args.sample_every))
        if ot is not None:
            print(f"cosine(swapped, other) = {cosine(sw, ot):.3f} "
                  f"(should be LOWER than to target)")


if __name__ == "__main__":
    main()
