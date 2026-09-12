"""
Optional headless/terminal entry point (plain OpenCV window, no
Tkinter) -- useful for quick testing or a machine without the
Tkinter dashboard set up. For the full monitoring GUI, use main.py.

Run:  python main_cli.py
Press 'q' to quit, 'r' to reset the protocol.
"""

from pathlib import Path

import cv2
import yaml

from capture.camera_stream import CameraStream
from event_logging.event_logger import EventLogger
from feedback.alert_manager import AlertManager
from pose.pose_estimator import PoseEstimator
from recognition.activity_classifier import classify as rule_based_classify
from recognition.model import ActivityModel
from streaming.video_streamer import LocalRecorder, NetworkStreamer
from validation.sequence_validator import SequenceValidator

BASE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = str(BASE_DIR / "config" / "protocol.yaml")
APP_CONFIG_PATH = str(BASE_DIR / "config" / "app_config.yaml")


def run():
    with open(APP_CONFIG_PATH) as f:
        app_config = yaml.safe_load(f)
    with open(PROTOCOL_PATH) as f:
        debounce_frames = yaml.safe_load(f).get("debounce_frames", 8)

    camera = CameraStream()
    pose_estimator = PoseEstimator()
    model = ActivityModel()
    validator = SequenceValidator(PROTOCOL_PATH)
    alerts = AlertManager()
    logger = EventLogger()
    recorder = LocalRecorder() if app_config.get("enable_recording", True) else None
    streamer = (
        NetworkStreamer(app_config["stream_ip"], app_config["stream_port"])
        if app_config.get("enable_streaming", True)
        else None
    )

    last_activity, stable_count = None, 0
    last_message = ("Starting...", (200, 200, 200))

    print(f"Expected sequence: {validator.steps}")
    print(f"Using trained model: {model.is_available}")
    if validator.steps:
        alerts.suggest_next(validator.steps[0])
    print("Press 'q' to quit, 'r' to reset the protocol.\n")

    try:
        while True:
            frame = camera.read_frame()
            if frame is None:
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks, results = pose_estimator.extract(frame_rgb)
            pose_estimator.draw_skeleton(frame, results)

            activity = model.predict(landmarks) if model.is_available else rule_based_classify(landmarks)

            if activity == last_activity:
                stable_count += 1
            else:
                stable_count, last_activity = 1, activity

            if stable_count == debounce_frames and activity not in ("no_person", "idle"):
                result = validator.check(activity)

                if result.status == "confirmed":
                    last_message = alerts.confirm(activity)
                    alerts.suggest_next(result.expected)
                elif result.status == "out_of_order":
                    last_message = alerts.warn(f"expected '{result.expected}', got '{activity}'")
                elif result.status == "unrecognized":
                    last_message = alerts.info(f"Detected '{activity}' (not part of protocol)")

                logger.log(activity, result.status)

            if recorder:
                recorder.write(frame)
            if streamer:
                streamer.send_frame(frame)

            progress = f"Step {validator.current_step}/{len(validator.steps)}"
            next_step = (
                validator.steps[validator.current_step]
                if validator.current_step < len(validator.steps)
                else "complete"
            )
            cv2.putText(frame, progress, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Next: {next_step}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            text, color = last_message
            cv2.putText(frame, text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("On-board HAR prototype (CLI)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                validator.reset()
                last_message = ("Protocol reset", (200, 200, 200))
                alerts.suggest_next(validator.steps[0] if validator.steps else None)

    finally:
        camera.release()
        pose_estimator.close()
        if recorder:
            recorder.release()
        if streamer:
            streamer.close()
        report_path = logger.write_summary(validator)
        print(f"Session summary written to {report_path}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
