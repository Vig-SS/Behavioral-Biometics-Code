# train_behavior_classifier.py  (paper version)
# Siddhi Satheesh — CSCI 693
# Project: Behavioral Biometrics for Distinguishing Visually Similar Users
#
# Same features and models as the original, refactored to import pose_common
# so capture/train/run/eval can never drift. Adds:
#   * --scores_out : writes a per-window CSV (user, recording_id, and the
#     predicted probability for every class) using GROUP-AWARE cross-validated
#     predictions. This is the file the evaluation + fusion scripts read.
#   * saves class order + metadata needed by the verification evaluation.
#
# Usage (closed-set sanity check + export scores):
#   python train_behavior_classifier.py --data_dir behavior_data \
#       --model_type logreg --model_out behavior_classifier.joblib \
#       --scores_out behavior_scores.csv

import os
import glob
import argparse

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GroupShuffleSplit, StratifiedKFold, StratifiedGroupKFold,
    cross_val_predict,
)
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
)

from pose_common import (
    METADATA_COLUMNS, extract_window_features_from_array, feature_column_names,
)


def extract_window_features(window_df):
    feature_df = window_df.drop(
        columns=[c for c in METADATA_COLUMNS if c in window_df.columns],
        errors="ignore",
    )
    values = feature_df.values.astype(np.float32)
    return extract_window_features_from_array(values)


def is_valid_pose_csv(df, file):
    if "user" not in df.columns:
        print(f"Skipping {file}: missing user column"); return False
    if "timestamp" not in df.columns:
        print(f"Skipping {file}: missing timestamp column"); return False
    if [c for c in df.columns if c.startswith("face_")]:
        print(f"Skipping {file}: old face+pose CSV detected"); return False
    feature_cols = [c for c in df.columns if c not in METADATA_COLUMNS]
    if len(feature_cols) == 0:
        print(f"Skipping {file}: no pose feature columns"); return False
    if len(feature_cols) % 4 != 0:
        print(f"Skipping {file}: feature count not divisible by 4"); return False
    return True


def load_dataset(data_dir, window_size, stride):
    X, y, groups = [], [], []
    feature_columns = None
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    print(f"Found {len(csv_files)} CSV files.")

    for file in csv_files:
        df = pd.read_csv(file)
        if not is_valid_pose_csv(df, file):
            continue
        current_cols = [c for c in df.columns if c not in METADATA_COLUMNS]
        if feature_columns is None:
            feature_columns = current_cols
        elif current_cols != feature_columns:
            print(f"Skipping {file}: columns don't match first CSV"); continue

        user_label = str(df["user"].iloc[0])
        recording_id = (str(df["recording_id"].iloc[0])
                        if "recording_id" in df.columns
                        else os.path.basename(file))

        for start in range(0, len(df) - window_size + 1, stride):
            window = df.iloc[start:start + window_size]
            if len(window) == window_size:
                X.append(extract_window_features(window))
                y.append(user_label)
                groups.append(recording_id)

    return (np.array(X), np.array(y), np.array(groups), feature_columns)


def make_classifier(model_type):
    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced")
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=42)),
    ])


def grouped_cv_scores(clf, X, y, groups, out_path):
    """Group-aware cross-validated class probabilities for every window."""
    n_groups = len(set(groups))
    n_classes = len(set(y))
    # choose a splitter that respects recording groups when possible
    if n_groups >= 4:
        n_splits = min(5, n_groups)
        try:
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                      random_state=42)
            splitter = cv.split(X, y, groups)
        except Exception:
            cv = GroupShuffleSplit(n_splits=n_splits, test_size=1.0 / n_splits,
                                   random_state=42)
            splitter = cv.split(X, y, groups)
        proba = cross_val_predict(clf, X, y, cv=list(splitter),
                                  method="predict_proba", groups=groups)
    else:
        cv = StratifiedKFold(n_splits=min(5, np.bincount(
            pd.factorize(y)[0]).min()), shuffle=True, random_state=42)
        proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")

    classes = sorted(set(y))
    # refit to learn class order used by predict_proba
    clf.fit(X, y)
    class_order = list(clf.classes_)

    rows = []
    for i in range(len(X)):
        row = {"user": y[i], "recording_id": groups[i]}
        for j, c in enumerate(class_order):
            row[f"score_{c}"] = float(proba[i, j])
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote per-window behavior scores -> {out_path} "
          f"({len(rows)} windows, {n_classes} classes)")
    return class_order


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="behavior_data")
    p.add_argument("--model_out", default="behavior_classifier.joblib")
    p.add_argument("--scores_out", default="behavior_scores.csv",
                   help="Per-window CV probability scores for eval/fusion")
    p.add_argument("--window_size", type=int, default=150,
                   help="Frames per sample (150 ~= 5s at 30 FPS)")
    p.add_argument("--stride", type=int, default=30)
    p.add_argument("--model_type", choices=["logreg", "rf"], default="logreg")
    args = p.parse_args()

    print("Loading pose-only dataset...")
    X, y, groups, feature_columns = load_dataset(
        args.data_dir, args.window_size, args.stride)
    if len(X) == 0:
        print("Error: No valid training samples found."); return

    print(f"Samples: {len(X)}   Feature size: {X.shape[1]}")
    print(f"Users: {sorted(set(y))}   Recordings: {len(set(groups))}")
    if len(set(y)) < 2:
        print("Error: Need at least 2 users to train."); return
    print("Samples per user:",
          {c: int(np.sum(y == c)) for c in sorted(set(y))})

    # ---- closed-set held-out sanity check (group split if possible) ----
    unique_groups = sorted(set(groups))
    if len(unique_groups) >= 4:
        tr, te = next(GroupShuffleSplit(
            n_splits=1, test_size=0.25, random_state=42
        ).split(X, y, groups=groups))
        print("Using group-based split by recording file.")
    else:
        from sklearn.model_selection import train_test_split
        idx = np.arange(len(X))
        tr, te = train_test_split(idx, test_size=0.25, random_state=42,
                                  stratify=y)
        print("Warning: few recordings; stratified split (optimistic).")

    clf = make_classifier(args.model_type)
    clf.fit(X[tr], y[tr])
    y_pred = clf.predict(X[te])
    print("\nAccuracy:", accuracy_score(y[te], y_pred))
    print("\nClassification Report:\n", classification_report(y[te], y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y[te], y_pred))

    # ---- export group-aware CV scores for the verification evaluation ----
    class_order = grouped_cv_scores(
        make_classifier(args.model_type), X, y, groups, args.scores_out)

    # ---- final model on all data + metadata ----
    final = make_classifier(args.model_type)
    final.fit(X, y)
    joblib.dump({
        "model": final,
        "window_size": args.window_size,
        "stride": args.stride,
        "feature_type": "upper_body_pose_only_normalized_motion_features",
        "feature_columns": feature_columns,
        "input_frame_features": len(feature_columns),
        "expected_input_features": X.shape[1],
        "model_type": args.model_type,
        "class_order": class_order,
    }, args.model_out)
    print(f"\nSaved model to: {args.model_out}")


if __name__ == "__main__":
    main()
