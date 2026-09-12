"""
Trains the offline activity-recognition model from labeled data
collected with recognition/data_collector.py, and saves it to
recognition/activity_model.pkl for model.py to load at runtime.

This is a plain scikit-learn RandomForestClassifier: small, fast to
train on CPU, needs no GPU/internet at train OR inference time, and
loads from a single pickle file -- appropriate for an "offline
standalone" deliverable.

Usage:
    python -m recognition.train_model
"""

import csv
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from recognition.features import FEATURE_NAMES

DATA_PATH = Path(__file__).resolve().parent / "training_data" / "dataset.csv"
MODEL_PATH = Path(__file__).resolve().parent / "activity_model.pkl"


def load_dataset(path=DATA_PATH):
    labels, features = [], []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            labels.append(row[0])
            features.append([float(v) for v in row[1:]])
    return np.array(features), np.array(labels)


def main():
    if not DATA_PATH.exists():
        raise SystemExit(
            f"No training data found at {DATA_PATH}.\n"
            "Run `python -m recognition.data_collector` first to record samples."
        )

    X, y = load_dataset()
    classes = sorted(set(y))
    print(f"Loaded {len(X)} samples across classes: {classes}")

    if len(X) < 20 or len(classes) < 2:
        raise SystemExit(
            "Not enough data to train yet -- record more bursts with "
            "data_collector.py for at least two different labels."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=150, random_state=42)
    clf.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, clf.predict(X_test))
    print(f"Held-out test accuracy: {accuracy:.2%}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": clf, "feature_names": FEATURE_NAMES}, f)
    print(f"Saved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
