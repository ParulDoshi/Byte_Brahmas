"""
Formats confirm/warning/info messages for on-screen display, AND
speaks the important ones aloud (voice-based alerts), so a step
skip or an out-of-sequence step doesn't require looking at a screen.

Text-to-speech runs on its own background thread with a small queue,
so a slow or stuck speech engine never blocks the video loop. If
pyttsx3 isn't installed, voice is silently disabled and everything
else keeps working.
"""

import queue
import threading

try:
    import pyttsx3
except ImportError:  # pragma: no cover - optional dependency
    pyttsx3 = None


class _VoiceWorker:
    def __init__(self):
        self._queue = queue.Queue()
        self._engine = pyttsx3.init() if pyttsx3 else None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        if not self._engine:
            return
        while True:
            text = self._queue.get()
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception:
                pass  # never let a TTS hiccup crash the app

    def speak(self, text):
        if self._engine:
            self._queue.put(text)


class AlertManager:
    def __init__(self, voice_enabled=True):
        self.voice_enabled = voice_enabled and pyttsx3 is not None
        self._voice = _VoiceWorker() if self.voice_enabled else None

    def _speak(self, text):
        if self._voice:
            self._voice.speak(text)

    def confirm(self, message):
        text = f"CONFIRMED: {message}"
        print(text)
        self._speak(f"{message} confirmed")
        return text, (0, 200, 0)  # green (BGR for OpenCV)

    def warn(self, message):
        text = f"WARNING: {message}"
        print(text)
        self._speak(f"Warning. {message}")
        return text, (0, 0, 255)  # red (BGR for OpenCV)

    def info(self, message):
        text = f"{message}"
        return text, (200, 200, 200)  # gray

    def suggest_next(self, step_name):
        """Voice-announce the next expected step (or completion)."""
        if step_name:
            self._speak(f"Next step: {step_name}")
        else:
            self._speak("Protocol complete")
