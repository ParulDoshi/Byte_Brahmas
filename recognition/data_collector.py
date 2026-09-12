"""
Interactive tool for recording labeled pose samples to train the
offline activity model. Perform each protocol step in front of the
webcam and press its key to capture a ~1 second burst of frames
labeled with that step name.

Usage:
    python -m recognition.data_collector

Repeat for every protocol step (and a few "idle" bursts) — a few
dozen bursts per label is enough for a hackathon-scale demo model.
Then run:
    python -m recognition.train_model
"""

import csv
import os
from pathlib import Path

import cv2
import yaml

from capture.camera_stream import CameraStream
from pose.pose_estimator import PoseEstimator
from recognition.features import to_feature_vector, FEATURE_NAMES

BASE_DIR = Path(__file__).resolve().parent.parent
PROTOCOL_PATH = BASE_DIR / "config" / "protocol.yaml"
DATA_PATH = Path(__file__).resolve().parent / "training_data" / "dataset.csv"
BURST_LENGTH = 30  # frames captured per key press (~1 second at 30fps)


def load_labels():
    with open(PROTOCOL_PATH) as f:
        steps = yaml.safe_load(f)["steps"]
    labels = {str(i): name for i, name in enumerate(steps)}
    labels["i"] = "idle"
    return labels


def ensure_dataset_file():
    os.makedirs(DATA_PATH.parent, exist_ok=True)
    if not DATA_PATH.exists():
        with open(DATA_PATH, "w", newline="") as f:
            csv.writer(f).writerow(["label", *FEATURE_NAMES])


def main():
    labels = load_labels()
    ensure_dataset_file()

    camera = CameraStream()
    pose_estimator = PoseEstimator()
    recording_label = None
    frames_left = 0

    hint = " | ".join(f"{k}={v}" for k, v in labels.items()) + " | q=quit"
    print("Perform each pose, then press its key to capture a burst of samples.")
    print(hint)

    try:
        while True:
            frame = camera.read_frame()
            if frame is None:
                break
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks, results = pose_estimator.extract(frame_rgb)
            pose_estimator.draw_skeleton(frame, results)

            if recording_label and frames_left > 0 and landmarks is not None:
                with open(DATA_PATH, "a", newline="") as f:
                    csv.writer(f).writerow([recording_label, *to_feature_vector(landmarks)])
                frames_left -= 1
                status = f"Recording '{recording_label}': {BURST_LENGTH - frames_left}/{BURST_LENGTH}"
            elif frames_left == 0:
                recording_label = None
                status = "Ready. " + hint
            else:
                status = "No person detected - recording paused"

            cv2.putText(frame, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.imshow("Data collector", frame)

            key = cv2.waitKey(1) & 0xFF
            key_char = chr(key) if key != 255 else ""
            if key_char == "q":
                break
            if key_char in labels and frames_left == 0:
                recording_label = labels[key_char]
                frames_left = BURST_LENGTH
    finally:
        camera.release()
        pose_estimator.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
