# behavior_to_comparisons.py
# Siddhi Satheesh — CSCI 693
# Project: Behavioral Biometrics for Distinguishing Visually Similar Users
#
# The behavior classifier outputs, per window, a probability for each user
# (behavior_scores.csv from train_behavior_classifier.py). To evaluate it the
# SAME way as the face baseline, we turn those probabilities into genuine /
# impostor COMPARISON scores:
#   for each window with true user u, and each candidate class c:
#       comparison score = P(class = c | window)
#       genuine = 1 if c == u else 0
# This yields a long table directly comparable to face_scores.csv.
#
# Usage:
#   python behavior_to_comparisons.py --scores behavior_scores.csv \
#       --out behavior_comparisons.csv

import argparse
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", default="behavior_scores.csv")
    p.add_argument("--out", default="behavior_comparisons.csv")
    args = p.parse_args()

    df = pd.read_csv(args.scores)
    score_cols = [c for c in df.columns if c.startswith("score_")]
    if not score_cols:
        print("No score_* columns found. Re-run training with --scores_out.")
        return
    classes = [c[len("score_"):] for c in score_cols]

    rows = []
    for _, r in df.iterrows():
        true_user = str(r["user"])
        rec = r.get("recording_id", "")
        for c, col in zip(classes, score_cols):
            rows.append({
                "probe_user": true_user,
                "recording_id": rec,
                "candidate_user": c,
                "behavior_score": float(r[col]),
                "genuine": int(c == true_user),
            })
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    n_gen = int(out["genuine"].sum())
    print(f"Wrote {len(out)} behavior comparisons -> {args.out} "
          f"({n_gen} genuine, {len(out) - n_gen} impostor)")


if __name__ == "__main__":
    main()
