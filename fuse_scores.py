# fuse_scores.py
# Siddhi Satheesh — CSCI 693
# Project: Behavioral Biometrics for Distinguishing Visually Similar Users
#
# Score-level fusion of the facial baseline and the behavioral classifier, the
# combination-of-techniques result the paper centers on. It aligns the two
# comparison tables on (probe_user, candidate_user), min-max normalizes each
# score, and fuses with a weighted sum. It then reports FAR/FRR/EER for face
# alone, behavior alone, and fused, so you can show fusion helps exactly on the
# similar-looking pairs.
#
# Inputs:
#   face_scores.csv        (from face_baseline.py):
#       probe_user, enrolled_user, face_score, genuine
#   behavior_comparisons.csv (from behavior_to_comparisons.py):
#       probe_user, candidate_user, behavior_score, genuine
#
# Because face is scored per probe CLIP and behavior per WINDOW, we aggregate
# behavior to the (probe_user, candidate_user) level (mean score) before fusing.
# For a stricter per-clip fusion, add matching clip ids to both tables.
#
# Usage:
#   python fuse_scores.py --face face_scores.csv \
#       --behavior behavior_comparisons.csv --w_face 0.5 --w_behavior 0.5 \
#       --target_far 0.01 --outdir results

import os
import argparse
import numpy as np
import pandas as pd

from verification_metrics import (
    far_frr_curve, equal_error_rate, threshold_at_far, rates_at_threshold,
)


def minmax(s):
    s = np.asarray(s, dtype=float)
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-12:
        return np.zeros_like(s)
    return (s - lo) / (hi - lo)


def summarize(name, scores, labels, target_far):
    thr, far, frr = far_frr_curve(scores, labels)
    eer, _ = equal_error_rate(thr, far, frr)
    t = threshold_at_far(thr, far, target_far)
    r = rates_at_threshold(scores, labels, t)
    return {
        "modality": name, "EER": round(eer, 4),
        "FAR@op": round(r["FAR"], 4), "FRR@op": round(r["FRR"], 4),
        "acc@op": round(r["accuracy"], 4), "threshold": round(t, 4),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--face", default="face_scores.csv")
    p.add_argument("--behavior", default="behavior_comparisons.csv")
    p.add_argument("--w_face", type=float, default=0.5)
    p.add_argument("--w_behavior", type=float, default=0.5)
    p.add_argument("--target_far", type=float, default=0.01)
    p.add_argument("--outdir", default="results")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    face = pd.read_csv(args.face).rename(
        columns={"enrolled_user": "candidate_user"})
    beh = pd.read_csv(args.behavior)

    # aggregate behavior to (probe_user, candidate_user)
    beh_agg = (beh.groupby(["probe_user", "candidate_user"])
               .agg(behavior_score=("behavior_score", "mean"),
                    genuine=("genuine", "max")).reset_index())
    face_agg = (face.groupby(["probe_user", "candidate_user"])
                .agg(face_score=("face_score", "mean"),
                     genuine=("genuine", "max")).reset_index())

    merged = pd.merge(face_agg, beh_agg,
                      on=["probe_user", "candidate_user", "genuine"],
                      how="inner")
    if merged.empty:
        print("No overlapping (probe_user, candidate_user) pairs between the "
              "two tables. Make sure the same user labels are used on both "
              "sides (face enroll/probe folders and behavior CSV 'user').")
        return

    merged["face_n"] = minmax(merged["face_score"])
    merged["beh_n"] = minmax(merged["behavior_score"])
    wsum = args.w_face + args.w_behavior
    merged["fused"] = (args.w_face * merged["face_n"] +
                       args.w_behavior * merged["beh_n"]) / wsum

    labels = merged["genuine"].values
    rows = [
        summarize("face", merged["face_n"].values, labels, args.target_far),
        summarize("behavior", merged["beh_n"].values, labels, args.target_far),
        summarize("fused", merged["fused"].values, labels, args.target_far),
    ]
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    table.to_csv(os.path.join(args.outdir, "fusion_summary.csv"), index=False)

    # per-pair: where does fusion rescue the impostor pairs face accepts?
    thr, far, frr = far_frr_curve(merged["face_n"].values, labels)
    t_face = threshold_at_far(thr, far, args.target_far)
    thr2, far2, frr2 = far_frr_curve(merged["fused"].values, labels)
    t_fused = threshold_at_far(thr2, far2, args.target_far)

    imp = merged[merged["genuine"] == 0].copy()
    imp["face_accepts"] = (imp["face_n"] >= t_face).astype(int)
    imp["fused_accepts"] = (imp["fused"] >= t_fused).astype(int)
    rescued = imp[(imp["face_accepts"] == 1) & (imp["fused_accepts"] == 0)]
    rescued[["probe_user", "candidate_user", "face_n", "beh_n", "fused"]].to_csv(
        os.path.join(args.outdir, "fusion_rescued_pairs.csv"), index=False)
    print(f"\nImpostor pairs face accepts but fusion rejects: {len(rescued)}")
    if len(rescued):
        print(rescued[["probe_user", "candidate_user"]].to_string(index=False))
    merged.to_csv(os.path.join(args.outdir, "fusion_merged_scores.csv"),
                  index=False)
    print(f"\nWrote fusion outputs to {args.outdir}/")


if __name__ == "__main__":
    main()
