# On-board HAR prototype (SIH demo)

A runnable prototype of the on-board Human Activity Recognition system:
webcam -> pose estimation -> trained activity classification -> sequence
validation against a defined experiment protocol -> voice/visual feedback
-> structured logging -> local recording + network streaming -> GUI.

## Setup

1. Python 3.9-3.11.
2. `pip install -r requirements.txt`
3. (Optional but recommended) train the real model -- see "Training the
   AI model" below. Until you do, the app runs on a rule-based fallback
   classifier so the pipeline works immediately.
4. Run the dashboard: `python main.py`
   - Or the lightweight terminal-only version: `python main_cli.py`

## What's new vs. the original pipeline demo

| Requirement | Where it lives |
|---|---|
| Suggests the next step | `SequenceValidator` tracks position; GUI/CLI display + speak it after every confirmed step |
| Voice alert on skip / out-of-sequence step | `feedback/alert_manager.py` (`pyttsx3` text-to-speech, background thread) |
| Timestamped structured text log of steps + outcomes | `event_logging/event_logger.py` -> `logs/session_log.csv`, `logs/session_log.jsonl`, and an end-of-session `logs/summary_*.txt` |
| Stream video to a specific IP + store locally | `streaming/video_streamer.py` (`NetworkStreamer` + `LocalRecorder`), configured in `config/app_config.yaml`; view the stream with `streaming/receiver_demo.py` |
| GUI for monitoring | `gui/app.py` -- Tkinter dashboard (video panel, current/next step, event log, reset/quit) |
| Trained AI model, offline/standalone | `recognition/features.py`, `data_collector.py`, `train_model.py`, `model.py` -- a scikit-learn `RandomForestClassifier` trained on your own recorded keypoint data, saved as a single `.pkl` file, no internet/GPU needed at inference time |

## Training the AI model

The model needs labeled examples of your real experiment steps before
it can replace the rule-based fallback:

1. Edit `config/protocol.yaml` with your real step names (or keep the
   demo poses to test the pipeline first).
2. Record samples:
   ```
   python -m recognition.data_collector
   ```
   Perform each step in front of the camera and press its number key
   (shown on screen) to capture a ~1 second burst of labeled frames.
   Repeat several times per step, from a few different angles/distances,
   plus a few bursts of `i` for "idle". Aim for 30-50+ bursts per label
   for a reasonable hackathon-scale model.
3. Train it:
   ```
   python -m recognition.train_model
   ```
   This prints held-out test accuracy and saves
   `recognition/activity_model.pkl`.
4. Run `python main.py` again -- it now uses the trained model
   automatically (the GUI's event log confirms "Using trained model: True").

## Streaming setup

Edit `config/app_config.yaml`:
```yaml
stream_ip: "127.0.0.1"   # the viewing machine's IP
stream_port: 9000
enable_streaming: true
enable_recording: true
```
On the viewing machine (or a second terminal for a one-laptop demo):
```
python streaming/receiver_demo.py --port 9000
```
Then start `python main.py` -- it connects and pushes the annotated
feed live. If no receiver is listening, streaming fails silently and
the rest of the app keeps running. Local recordings always save to
`recordings/session_<timestamp>.mp4` regardless of streaming status.

## Putting this on GitHub

Your installed environment (mediapipe, opencv, scikit-learn, etc.) can
easily reach 1-2 GB on disk. Never commit that -- push only source
code, config, and `requirements.txt`; everyone who clones the repo
recreates the environment themselves. This repo already ships a
`.gitignore` that excludes the venv folder and any runtime-generated
files (`logs/`, `recordings/`, trained model, collected training CSVs).

```
git init
git add .
git commit -m "Initial commit: on-board HAR prototype"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

If you already ran `git add .` once before adding `.gitignore` and a
huge venv folder got committed, untrack it without deleting it from
disk:
```
git rm -r --cached <venv-folder-name>
git commit -m "Stop tracking the virtual environment"
```
GitHub also hard-rejects any single file over 100 MB, so if a push
fails with that error, it's almost always the venv or a recorded
`.mp4` sneaking in -- check `.gitignore` covers it.

If you want to include a demo video for judges, don't commit it to
the repo; attach it as a GitHub Release asset, or link a
Drive/YouTube upload from the README instead.

## Installation troubleshooting

Read this before your demo, not during it -- most of these have
already bitten this exact project during development.

**Confirm which Python you're actually installing into.** The single
most common failure mode is installing packages into one Python and
running the app with another. Always verify with:
```
python -c "import sys; print(sys.executable)"
```
and make sure that path is inside the venv you intend to use. If in
doubt, install explicitly with that interpreter:
```
<path-to-venv>\Scripts\python.exe -m pip install -r requirements.txt
```

**`ModuleNotFoundError: No module named 'mediapipe'`** (or any other
package) — the install landed in a different interpreter than the one
running `main.py`. Reinstall using the exact interpreter path above,
not a bare `pip install`.

**`AttributeError: module 'mediapipe' has no attribute 'solutions'`**
— a real regression in newer MediaPipe releases (0.10.30+ and the 1.x
series) that dropped/broke the legacy `solutions` API many projects
still use. `requirements.txt` already pins `mediapipe==0.10.21`,
the last known-good release. If you ever bump the version, revert to
this pin if the error comes back:
```
pip uninstall mediapipe -y
pip install mediapipe==0.10.21
```

**`ERROR: No matching distribution found for mediapipe`** — usually
means an unsupported Python version/architecture. MediaPipe ships
wheels for Python 3.9-3.11, 64-bit only. Check both:
```
python -c "import struct,sys; print(sys.version); print(struct.calcsize('P')*8, 'bit')"
```
If you see 32-bit or a Python version outside 3.9-3.11, install a
64-bit Python 3.11 from python.org and recreate the venv with it.

**`TypeError: Descriptors cannot be created directly` / other
protobuf errors** — a version conflict between mediapipe's required
`protobuf` and one pulled in by another package. Let mediapipe manage
its own protobuf version rather than pinning one yourself:
```
pip uninstall protobuf -y
pip install mediapipe==0.10.21
```

**`ImportError: DLL load failed while importing cv2` (Windows)** —
missing C++ runtime. Install the "Microsoft Visual C++ Redistributable
for Visual Studio 2015-2022" (x64) from Microsoft's site, then retry.

**`error: Microsoft Visual C++ 14.0 or greater is required`** — pip is
trying to build a package from source instead of using a prebuilt
wheel, almost always because your Python version/architecture doesn't
have a matching wheel available. Switch to a supported Python version
(3.10 or 3.11, 64-bit) rather than installing build tools.

**`error: externally-managed-environment` (Linux, pip 23.x+)** — your
system Python is protected from direct pip installs. Either use a venv
(recommended, and what this project assumes), or if you must install
globally: `pip install --break-system-packages -r requirements.txt`.

**`ModuleNotFoundError: No module named 'tkinter'` (Linux only)** —
Tkinter isn't part of the pip-installable packages; install it via
your OS package manager: `sudo apt install python3-tk` (Debian/Ubuntu)
or the equivalent for your distro. Windows/Mac installs from
python.org include it already.

**No voice output from alerts** — `pyttsx3` needs a platform TTS
backend. Windows (SAPI5) and macOS (NSSpeechSynthesizer) work out of
the box. On Linux, install `espeak` first: `sudo apt install espeak`.
If you still hear nothing, the app keeps running silently -- voice is
never required for the app to function, just the alert text.

**Webcam window is black, or `camera.read_frame()` returns nothing** —
another app (Zoom, Teams, browser tab) is holding the camera; close
it. On Windows, also check Settings -> Privacy & security -> Camera ->
"Let desktop apps access your camera" is on. If you have multiple
cameras, try `CameraStream(1)` instead of the default index `0` in
`capture/camera_stream.py`.

**`python -m venv` fails / venv module missing (Linux)** — install it
first: `sudo apt install python3-venv`, then retry venv creation.

**Multiple Python versions installed on Windows and unsure which
you're getting** — use the launcher to pick explicitly:
```
py -3.11 -m venv prototype
prototype\Scripts\activate
python -m pip install -r requirements.txt
```

**Quick end-to-end sanity check**, once installed, before touching the
GUI/camera at all:
```
python -c "import cv2, mediapipe, yaml, numpy, PIL, sklearn; print(mediapipe.__version__); print(hasattr(mediapipe,'solutions'))"
```
Should print `0.10.21` and `True` with no errors.

## Project layout

```
har_bas_prototype/
├── config/
│   ├── protocol.yaml            # expected step sequence + debounce
│   └── app_config.yaml          # streaming/recording settings
├── capture/camera_stream.py     # webcam wrapper
├── pose/pose_estimator.py       # MediaPipe pose -> keypoints
├── recognition/
│   ├── features.py              # landmarks -> fixed feature vector
│   ├── data_collector.py        # record labeled training samples
│   ├── train_model.py           # trains + saves activity_model.pkl
│   ├── model.py                 # loads + runs the trained model
│   └── activity_classifier.py   # rule-based fallback (pre-training)
├── validation/sequence_validator.py  # FSM checking protocol order
├── feedback/alert_manager.py    # on-screen + VOICE alerts
├── event_logging/event_logger.py     # CSV + JSONL + summary report
├── streaming/
│   ├── video_streamer.py        # network push + local recording
│   └── receiver_demo.py         # standalone viewer for the stream
├── gui/app.py                   # Tkinter monitoring dashboard
├── main.py                      # GUI entry point
└── main_cli.py                  # headless/terminal entry point
```

## Known limitations (worth mentioning if you present this)

- The shipped `activity_model.pkl` doesn't exist until *you* record
  data and train it -- the demo poses in `protocol.yaml` are a
  convenient starting point, but real accuracy depends on your own
  recorded samples.
- No multi-person handling -- assumes one person in frame.
- `NetworkStreamer` is a simple length-prefixed TCP/JPEG protocol for
  demo purposes, not a production video protocol (no encryption, no
  reconnection backoff beyond retry-on-next-frame).
- Voice alerts require a working audio output device and `pyttsx3`'s
  platform TTS backend (SAPI5 on Windows, works out of the box).
