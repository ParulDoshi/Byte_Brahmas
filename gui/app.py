"""
Tkinter dashboard for the on-board HAR prototype -- the "GUI for
monitoring" deliverable. Shows the live annotated camera feed plus
the current step, the suggested next step, and a scrolling event/
alert log, and wires together every other piece (trained model,
sequence validation, voice alerts, structured logging, local
recording, and network streaming).
"""

import tkinter as tk
from datetime import datetime
from tkinter import ttk

import cv2
import yaml
from PIL import Image, ImageTk

from capture.camera_stream import CameraStream
from event_logging.event_logger import EventLogger
from feedback.alert_manager import AlertManager
from pose.pose_estimator import PoseEstimator
from recognition.activity_classifier import classify as rule_based_classify
from recognition.model import ActivityModel
from streaming.video_streamer import LocalRecorder, NetworkStreamer
from validation.sequence_validator import SequenceValidator


class HARApp:
    def __init__(self, root, protocol_path, app_config_path):
        self.root = root
        self.root.title("On-board HAR - Mission Monitor")

        with open(app_config_path) as f:
            self.app_config = yaml.safe_load(f)
        with open(protocol_path) as f:
            self.debounce_frames = yaml.safe_load(f).get("debounce_frames", 8)

        self.camera = CameraStream()
        self.pose_estimator = PoseEstimator()
        self.model = ActivityModel()  # falls back to rule-based if not yet trained
        self.validator = SequenceValidator(protocol_path)
        self.alerts = AlertManager()
        self.logger = EventLogger()

        self.recorder = (
            LocalRecorder() if self.app_config.get("enable_recording", True) else None
        )
        self.streamer = (
            NetworkStreamer(self.app_config["stream_ip"], self.app_config["stream_port"])
            if self.app_config.get("enable_streaming", True)
            else None
        )

        self.last_activity = None
        self.stable_count = 0
        self.running = True

        self._build_layout()
        self._log(f"Using trained model: {self.model.is_available}")
        self._announce_next_step(initial=True)
        self._update_loop()

    # ---------- UI construction ----------
    def _build_layout(self):
        video_frame = ttk.Frame(self.root)
        video_frame.grid(row=0, column=0, padx=8, pady=8)
        self.video_label = ttk.Label(video_frame)
        self.video_label.pack()

        panel = ttk.Frame(self.root)
        panel.grid(row=0, column=1, sticky="n", padx=8, pady=8)

        self.step_var = tk.StringVar()
        self.next_var = tk.StringVar()
        self.activity_var = tk.StringVar(value="-")

        ttk.Label(panel, text="Current progress", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(panel, textvariable=self.step_var).pack(anchor="w", pady=(0, 8))

        ttk.Label(panel, text="Suggested next step", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(panel, textvariable=self.next_var, foreground="#0a7a3d").pack(anchor="w", pady=(0, 8))

        ttk.Label(panel, text="Detected activity", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(panel, textvariable=self.activity_var).pack(anchor="w", pady=(0, 8))

        ttk.Label(panel, text="Event log", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.log_box = tk.Text(panel, width=44, height=18, state="disabled")
        self.log_box.pack()

        btns = ttk.Frame(panel)
        btns.pack(pady=8, fill="x")
        ttk.Button(btns, text="Reset protocol", command=self._reset).pack(side="left")
        ttk.Button(btns, text="Quit", command=self._quit).pack(side="right")

        self._refresh_progress_label()

    # ---------- helpers ----------
    def _refresh_progress_label(self):
        self.step_var.set(f"Step {self.validator.current_step}/{len(self.validator.steps)}")

    def _log(self, text):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"[{timestamp}] {text}\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _announce_next_step(self, initial=False):
        upcoming = (
            self.validator.steps[self.validator.current_step]
            if self.validator.current_step < len(self.validator.steps)
            else None
        )
        self.next_var.set(upcoming or "Protocol complete")
        self.alerts.suggest_next(upcoming)
        self._log(f"{'Start' if initial else 'Next'} step suggested: {upcoming or 'none - complete'}")

    def _reset(self):
        self.validator.reset()
        self.stable_count = 0
        self.last_activity = None
        self._refresh_progress_label()
        self._announce_next_step()
        self._log("Protocol reset")

    def _quit(self):
        self.running = False
        self.root.after(50, self._shutdown)

    def _shutdown(self):
        self.camera.release()
        self.pose_estimator.close()
        if self.recorder:
            self.recorder.release()
        if self.streamer:
            self.streamer.close()
        report_path = self.logger.write_summary(self.validator)
        print(f"Session summary written to {report_path}")
        self.root.destroy()

    # ---------- main loop ----------
    def _update_loop(self):
        if not self.running:
            return

        frame = self.camera.read_frame()
        if frame is not None:
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks, results = self.pose_estimator.extract(frame_rgb)
            self.pose_estimator.draw_skeleton(frame, results)

            activity = (
                self.model.predict(landmarks)
                if self.model.is_available
                else rule_based_classify(landmarks)
            )
            self.activity_var.set(activity)

            if activity == self.last_activity:
                self.stable_count += 1
            else:
                self.stable_count = 1
                self.last_activity = activity

            if self.stable_count == self.debounce_frames and activity not in ("no_person", "idle"):
                result = self.validator.check(activity)

                if result.status == "confirmed":
                    self.alerts.confirm(activity)
                    self._log(f"CONFIRMED: {activity}")
                    self._refresh_progress_label()
                    self._announce_next_step()
                elif result.status == "out_of_order":
                    msg = f"expected '{result.expected}', got '{activity}'"
                    self.alerts.warn(msg)
                    self._log(f"OUT OF SEQUENCE: {msg}")
                elif result.status == "unrecognized":
                    self._log(f"Unrecognized activity: {activity}")

                self.logger.log(activity, result.status)

            if self.recorder:
                self.recorder.write(frame)
            if self.streamer:
                self.streamer.send_frame(frame)

            display_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(image=Image.fromarray(display_rgb))
            self.video_label.imgtk = img  # keep a reference so it isn't garbage collected
            self.video_label.configure(image=img)

        self.root.after(15, self._update_loop)


def run(protocol_path, app_config_path):
    root = tk.Tk()
    app = HARApp(root, protocol_path, app_config_path)
    root.protocol("WM_DELETE_WINDOW", app._quit)
    root.mainloop()
