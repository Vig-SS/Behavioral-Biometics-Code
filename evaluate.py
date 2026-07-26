# evaluate.py
# Siddhi Satheesh — CSCI 693
# Project: Behavioral Biometrics for Distinguishing Visually Similar Users
#
# Produces the paper's core numbers for ONE modality at a time (face OR
# behavior), plus the similar-looking-pair breakdown that is the whole point:
# which specific user pairs get confused, and at what rate.
#
# Reads a comparison table with columns:
#   <score_col>, genuine (1/0), and identity columns to name the pairs.
#
# Outputs (into --outdir):
#   <tag>_summary.txt         accuracy/EER/FAR/FRR at operating point
#   <tag>_far_frr.csv         full threshold sweep
#   <tag>_roc.png / det.png   curves (if matplotlib present)
#   <tag>_confusion.csv       closed-set confusion (behavior only, if scores)
#   <tag>_pair_far.csv        impostor acceptance rate per (probe,candidate) pair
#
# Usage examples:
#   python evaluate.py --comparisons behavior_comparisons.csv \
#       --score_col behavior_score --probe_col probe_user \
#       --candidate_col candidate_user --tag behavior --target_far 0.01
#
#   python evaluate.py --comparisons face_scores.csv \
#       --score_col face_score --probe_col probe_user \
#       --candidate_col enrolled_user --tag face --target_far 0.01

import os
import argparse
import numpy as np
import pandas as pd

from verification_metrics import (
    far_frr_curve, equal_error_rate, threshold_at_far, rates_at_threshold,
    auc_roc,
)


def maybe_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--comparisons", required=True)
    p.add_argument("--score_col", required=True)
    p.add_argument("--probe_col", required=True)
    p.add_argument("--candidate_col", required=True)
    p.add_argument("--tag", default="model")
    p.add_argument("--target_far", type=float, default=0.01)
    p.add_argument("--outdir", default="results")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.comparisons)
    scores = df[args.score_col].values
    labels = df["genuine"].values

    thr, far, frr = far_frr_curve(scores, labels)
    tar = 1.0 - frr
    eer, eer_t = equal_error_rate(thr, far, frr)
    op_t = threshold_at_far(thr, far, args.target_far)
    op = rates_at_threshold(scores, labels, op_t)
    roc_auc = auc_roc(far, tar)

    # ---- summary ----
    lines = [
        f"Modality: {args.tag}",
        f"Comparisons: {len(df)}  (genuine={int(labels.sum())}, "
        f"impostor={int((labels == 0).sum())})",
        f"ROC AUC: {roc_auc:.4f}",
        f"EER: {eer:.4f} at threshold {eer_t:.4f}",
        f"Operating point (target FAR <= {args.target_far}):",
        f"  threshold = {op['threshold']:.4f}",
        f"  FAR = {op['FAR']:.4f}",
        f"  FRR = {op['FRR']:.4f}",
        f"  accuracy = {op['accuracy']:.4f}",
    ]
    summary = "\n".join(lines)
    print(summary)
    with open(os.path.join(args.outdir, f"{args.tag}_summary.txt"), "w") as f:
        f.write(summary + "\n")

    # ---- sweep csv ----
    pd.DataFrame({"threshold": thr, "FAR": far, "FRR": frr, "TAR": tar}).to_csv(
        os.path.join(args.outdir, f"{args.tag}_far_frr.csv"), index=False)

    # ---- per-pair impostor acceptance (the similar-looking pair story) ----
    imp = df[df["genuine"] == 0].copy()
    imp["accepted"] = (imp[args.score_col] >= op_t).astype(int)
    pair = (imp.groupby([args.probe_col, args.candidate_col])["accepted"]
            .agg(["mean", "count"]).reset_index()
            .rename(columns={"mean": "pair_FAR", "count": "n"}))
    pair = pair.sort_values("pair_FAR", ascending=False)
    pair.to_csv(os.path.join(args.outdir, f"{args.tag}_pair_far.csv"),
                index=False)
    if len(pair):
        print("\nMost-confused impostor pairs (at operating point):")
        print(pair.head(10).to_string(index=False))

    # ---- closed-set confusion (argmax over candidates per probe row group) --
    # Reconstruct a per-probe decision: for each probe instance, pick the
    # candidate with the max score. Works when every probe is scored against
    # all candidates (the behavior + face tables both do this).
    key_cols = [args.probe_col]
    if "recording_id" in df.columns:
        key_cols.append("recording_id")
    if "probe_clip" in df.columns:
        key_cols.append("probe_clip")
    try:
        idx = df.groupby(key_cols)[args.score_col].idxmax()
        decided = df.loc[idx]
        y_true = decided[args.probe_col].astype(str)
        y_pred = decided[args.candidate_col].astype(str)
        classes = sorted(set(y_true) | set(y_pred))
        # Use fixed Categorical categories so the matrix is always len(classes)
        # x len(classes), even when EVERY prediction collapses to one identity
        # (which is exactly what a fooled face matcher does on swapped clips).
        yt = pd.Categorical(y_true, categories=classes)
        yp = pd.Categorical(y_pred, categories=classes)
        cm = pd.crosstab(yt, yp, dropna=False)
        cm.index.name = "true"; cm.columns.name = "pred"
        cm.to_csv(os.path.join(args.outdir, f"{args.tag}_confusion.csv"))
        acc = float((y_true.values == y_pred.values).mean())
        n_pred_classes = y_pred.nunique()
        print(f"\nClosed-set identification accuracy: {acc:.4f}")
        if n_pred_classes == 1:
            print(f"  NOTE: every probe was matched to a single identity "
                  f"('{y_pred.iloc[0]}') — the matcher cannot separate these "
                  f"clips at all (expected for fooled look-alikes).")
        with open(os.path.join(args.outdir, f"{args.tag}_summary.txt"),
                  "a") as f:
            f.write(f"Closed-set identification accuracy: {acc:.4f}\n")
            if n_pred_classes == 1:
                f.write("NOTE: all probes collapsed to one identity "
                        f"('{y_pred.iloc[0]}').\n")
    except Exception as e:
        print("Confusion step skipped:", e)

    # ---- plots ----
    plt = maybe_plt()
    if plt is not None:
        plt.figure()
        plt.plot(far, tar)
        plt.xlabel("False Acceptance Rate"); plt.ylabel("True Acceptance Rate")
        plt.title(f"ROC — {args.tag} (AUC={roc_auc:.3f})")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(args.outdir, f"{args.tag}_roc.png"),
                    dpi=140, bbox_inches="tight"); plt.close()

        plt.figure()
        plt.plot(far, frr)
        plt.scatter([eer], [eer], zorder=5)
        plt.xlabel("FAR"); plt.ylabel("FRR")
        plt.title(f"DET — {args.tag} (EER={eer:.3f})")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(args.outdir, f"{args.tag}_det.png"),
                    dpi=140, bbox_inches="tight"); plt.close()
        print(f"\nSaved ROC/DET plots to {args.outdir}/")
    else:
        print("\n(matplotlib not installed — skipped plots; CSVs still written)")


if __name__ == "__main__":
    main()
