# verification_metrics.py
# Siddhi Satheesh — CSCI 693
# Project: Behavioral Biometrics for Distinguishing Visually Similar Users
#
# Verification metrics shared by the evaluation and fusion scripts. Everything
# here treats identification as a VERIFICATION problem (genuine vs impostor
# comparison scores), which is what the paper's FAR/FRR framing needs.

import numpy as np


def far_frr_curve(scores, labels, num_thresholds=200):
    """
    scores: array of comparison scores (higher = more likely genuine)
    labels: 1 = genuine comparison, 0 = impostor comparison
    Returns thresholds, FAR(t), FRR(t).
      FAR = impostors accepted / impostors
      FRR = genuines rejected / genuines
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    lo, hi = scores.min(), scores.max()
    if lo == hi:
        lo, hi = lo - 1e-6, hi + 1e-6
    thr = np.linspace(lo, hi, num_thresholds)

    gen = scores[labels == 1]
    imp = scores[labels == 0]
    far, frr = [], []
    for t in thr:
        # accept if score >= t
        far.append(np.mean(imp >= t) if len(imp) else 0.0)
        frr.append(np.mean(gen < t) if len(gen) else 0.0)
    return thr, np.array(far), np.array(frr)


def equal_error_rate(thr, far, frr):
    """EER = point where FAR ~= FRR. Returns (eer, threshold)."""
    diff = np.abs(far - frr)
    i = int(np.argmin(diff))
    return float((far[i] + frr[i]) / 2.0), float(thr[i])


def threshold_at_far(thr, far, target_far):
    """Largest threshold achieving FAR <= target (operating point)."""
    ok = np.where(far <= target_far)[0]
    if len(ok) == 0:
        return float(thr[-1])
    # far is non-increasing in t; pick the smallest threshold meeting target
    return float(thr[ok[0]])


def rates_at_threshold(scores, labels, t):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    gen = scores[labels == 1]
    imp = scores[labels == 0]
    far = float(np.mean(imp >= t)) if len(imp) else 0.0
    frr = float(np.mean(gen < t)) if len(gen) else 0.0
    acc = float(np.mean((scores >= t).astype(int) == labels))
    return {"threshold": t, "FAR": far, "FRR": frr, "accuracy": acc}


def auc_roc(far, tar):
    """Area under ROC given FAR (x) and TAR=1-FRR (y). Sorted by FAR."""
    order = np.argsort(far)
    x = np.array(far)[order]
    y = np.array(tar)[order]
    trap = getattr(np, "trapezoid", None) or np.trapz  # numpy>=2 renamed it
    return float(trap(y, x))
