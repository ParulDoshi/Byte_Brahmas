"""
Wraps MediaPipe Pose. Turns a raw camera frame into a dictionary of
body joint positions (a "skeleton") that the activity classifier can
reason about, instead of raw pixels.
"""

import mediapipe as mp


class PoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def extract(self, frame_rgb):
        """
        Returns (landmarks_dict, raw_results) where landmarks_dict maps
        joint name -> (x, y, visibility), with x/y normalized 0-1.
        Returns (None, None) if no person is detected in the frame.
        """
        results = self.pose.process(frame_rgb)
        if not results.pose_landmarks:
            return None, None

        landmarks = {}
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            name = self.mp_pose.PoseLandmark(idx).name
            landmarks[name] = (lm.x, lm.y, lm.visibility)

        return landmarks, results

    def draw_skeleton(self, frame, results):
        """Draws the detected skeleton onto the frame for visual feedback."""
        if results and results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )

    def close(self):
        self.pose.close()
