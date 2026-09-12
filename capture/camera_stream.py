"""
Wraps the webcam so the rest of the code doesn't need to know
anything about OpenCV's VideoCapture API directly.
"""

import cv2


class CameraStream:
    def __init__(self, camera_index: int = 0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                "Could not open the camera. Check that it isn't being used "
                "by another app, or try camera_index=1 if you have more "
                "than one camera."
            )

    def read_frame(self):
        """Returns a single BGR frame, or None if the camera has stopped."""
        success, frame = self.cap.read()
        if not success:
            return None
        return frame

    def release(self):
        self.cap.release()
