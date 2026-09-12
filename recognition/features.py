"""
Shared feature extraction: turns a MediaPipe landmarks dict into a
fixed-length numeric vector, so the same representation is used both
when collecting training data (data_collector.py) and when running
the trained model live (model.py).
"""

# Order matches mediapipe.solutions.pose.PoseLandmark (indices 0-32).
# Duplicated here (rather than imported from mediapipe) so this module
# has no mediapipe dependency of its own.
LANDMARK_ORDER = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
    "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT",
    "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY",
    "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB",
    "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
]

FEATURE_NAMES = [f"{name}_{axis}" for name in LANDMARK_ORDER for axis in ("x", "y", "v")]


def to_feature_vector(landmarks):
    """
    landmarks: dict from PoseEstimator.extract(), keyed by joint name
    to (x, y, visibility), or None if no person was detected.

    Returns a flat list of 99 floats (33 joints x, y, visibility), or
    None if no person was detected.
    """
    if landmarks is None:
        return None
    return [value for name in LANDMARK_ORDER for value in landmarks[name]]
