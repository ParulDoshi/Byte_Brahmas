"""
Handles the two video-output requirements:
  - LocalRecorder:   saves the annotated session video to disk.
  - NetworkStreamer: pushes JPEG frames over TCP to a configured
                      IP:port so a "mission control" viewer
                      (see receiver_demo.py) can watch the feed live.

NetworkStreamer is best-effort and runs on its own thread: if no
receiver is listening at the target IP, it drops frames silently
rather than blocking or crashing the main capture loop.
"""

import os
import queue
import socket
import struct
import threading
from datetime import datetime

import cv2


class LocalRecorder:
    def __init__(self, output_dir="recordings", fps=20):
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(output_dir, f"session_{timestamp}.mp4")
        self.fps = fps
        self.writer = None  # created lazily once we know the frame size

    def write(self, frame):
        if self.writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(self.path, fourcc, self.fps, (w, h))
        self.writer.write(frame)

    def release(self):
        if self.writer is not None:
            self.writer.release()


class NetworkStreamer:
    """
    Best-effort TCP frame pusher. Each frame is sent as a 4-byte
    big-endian length prefix followed by that many bytes of
    JPEG-encoded image data, so receiver_demo.py can read frames
    back out one at a time.
    """

    def __init__(self, target_ip, target_port, jpeg_quality=70):
        self.target = (target_ip, target_port)
        self.jpeg_quality = jpeg_quality
        self._queue = queue.Queue(maxsize=2)  # drop old frames, stay live
        self._sock = None
        self._connected = False
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _connect(self):
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(2)
            self._sock.connect(self.target)
            self._connected = True
            print(f"[stream] connected to {self.target}")
        except OSError:
            self._sock = None
            self._connected = False

    def _run(self):
        while not self._stop:
            frame = self._queue.get()
            if frame is None:
                continue
            if not self._connected:
                self._connect()
                if not self._connected:
                    continue  # drop this frame, retry on the next one
            try:
                ok, buf = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                )
                if not ok:
                    continue
                data = buf.tobytes()
                self._sock.sendall(struct.pack(">I", len(data)) + data)
            except OSError:
                self._connected = False
                self._sock = None

    def send_frame(self, frame):
        if self._queue.full():
            try:
                self._queue.get_nowait()  # drop the oldest queued frame
            except queue.Empty:
                pass
        self._queue.put(frame)

    def close(self):
        self._stop = True
        self._queue.put(None)
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
