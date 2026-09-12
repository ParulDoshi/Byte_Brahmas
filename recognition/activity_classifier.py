"""
This is a RULE-BASED FALLBACK, used automatically only until you've
trained the real model (see data_collector.py + train_model.py +
model.py). It looks at one frame's joint positions and guesses the
activity from simple geometry -- no training data required, so the
full pipeline is demoable immediately, before any data collection.

Once recognition/activity_model.pkl exists, main.py / main_cli.py /
gui/app.py use ActivityModel.predict() instead of this function.
"""


def classify(landmarks):
    """
    landmarks: dict from PoseEstimator.extract(), or None if no person
    was detected. Returns a string activity label.

    Note: in image coordinates, y increases DOWNWARD, so "higher up"
    means a SMALLER y value.
    """
    if landmarks is None:
        return "no_person"

    r_wrist = landmarks["RIGHT_WRIST"]
    l_wrist = landmarks["LEFT_WRIST"]
    r_shoulder = landmarks["RIGHT_SHOULDER"]
    l_hip = landmarks["LEFT_HIP"]

    hand_distance = (
        (r_wrist[0] - l_wrist[0]) ** 2 + (r_wrist[1] - l_wrist[1]) ** 2
    ) ** 0.5

    if r_wrist[1] < r_shoulder[1] - 0.05:
        return "raise_hand"
    if hand_distance < 0.08:
        return "hands_together"
    if r_wrist[1] > l_hip[1] + 0.05:
        return "reach_down"
    return "idle"
