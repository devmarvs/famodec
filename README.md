# Face & Motion Detection

Lightweight OpenCV tools for detecting faces and motion from a camera feed, plus a simple Flask web UI to view the stream in a browser.

## Requirements
- Python 3.9+ recommended
- Packages: `opencv-python` (or `opencv-python-headless` if you do not need windows), `flask` for the web UI

```bash
python -m pip install opencv-python flask
# or headless: python -m pip install opencv-python-headless flask
```

## CLI detector (desktop window)
Script: `motion.py`

```bash
python motion_detect/motion.py --source 0
# quit with q or Esc
```

- Green boxes label faces; blue boxes mark motion regions.
- Useful flags:
  - `--scale` (default 1.2) change face sensitivity (lower is more sensitive).
  - `--neighbors` (default 5) adjust strictness for face detection.
  - `--min-size` minimum face size in pixels (default 40).
  - `--motion` / `--no-motion` toggle motion overlay (default on).
  - `--min-motion-area` minimum moving area in pixels (default 500).
  - `--source` camera index (0/1/2) or video path.

## Web UI (browser stream)
Script: `web_app.py`

```bash
python motion_detect/web_app.py --source 0 --port 5000
# open http://localhost:5000
```

- Serves MJPEG frames with the same overlays.
- Keep the terminal open while streaming; Ctrl+C to stop.

## Tips
- If you have multiple cameras, try `--source 1` or `--source 2`.
- For noisy scenes, raise `--min-motion-area`; for more motion sensitivity, lower it.
- On headless servers, use the web UI and install `opencv-python-headless`.***
