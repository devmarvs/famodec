import argparse
import sys
from pathlib import Path

import cv2
from flask import Flask, Response, render_template_string


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Web UI for live face + motion detection.")
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index (e.g. 0) or path to a video file. Defaults to the first camera.",
    )
    parser.add_argument("--port", type=int, default=5000, help="Port to run the web server on.")
    parser.add_argument(
        "--scale",
        type=float,
        default=1.2,
        help="How much the image size is reduced at each image scale for face detection.",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=5,
        help="How many neighbors each candidate rectangle should have to retain it.",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=40,
        help="Minimum face size in pixels to detect.",
    )
    parser.add_argument(
        "--min-motion-area",
        type=int,
        default=500,
        help="Minimum contour area to treat as motion (in pixels).",
    )
    return parser.parse_args()


def load_classifier() -> cv2.CascadeClassifier:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    classifier = cv2.CascadeClassifier(str(cascade_path))
    if classifier.empty():
        raise RuntimeError(f"Failed to load cascade from {cascade_path}")
    return classifier


class Detector:
    """Lightweight face + motion detector."""

    def __init__(self, classifier: cv2.CascadeClassifier, scale: float, neighbors: int, min_size: int, min_motion_area: int):
        self.classifier = classifier
        self.scale = scale
        self.neighbors = neighbors
        self.min_size = min_size
        self.min_motion_area = min_motion_area
        self.prev_gray = None

    def annotate(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_boxes = self._detect_motion(gray)

        faces = self.classifier.detectMultiScale(
            gray,
            scaleFactor=self.scale,
            minNeighbors=self.neighbors,
            minSize=(self.min_size, self.min_size),
        )

        for idx, (x, y, w, h) in enumerate(faces, start=1):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Face {idx}",
                (x, y - 10 if y - 10 > 10 else y + h + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        for (x, y, w, h) in motion_boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.putText(
            frame,
            f"Faces: {len(faces)} | Motion: {len(motion_boxes)}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )
        return frame

    def _detect_motion(self, gray):
        motion_boxes = []
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        if self.prev_gray is None:
            self.prev_gray = blurred
            return motion_boxes

        delta = cv2.absdiff(self.prev_gray, blurred)
        thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < self.min_motion_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            motion_boxes.append((x, y, w, h))
        self.prev_gray = blurred
        return motion_boxes


def resolve_source(source: str):
    if source.isdigit():
        return int(source)
    return source


def create_app(detector: Detector, capture: cv2.VideoCapture) -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(
            """
            <!doctype html>
            <html lang="en">
            <head>
                <meta charset="utf-8">
                <title>Face + Motion Stream</title>
                <style>
                    body { font-family: Arial, sans-serif; background: #0b1c2c; color: #e8f1f8; text-align: center; }
                    img { width: 90vw; max-width: 960px; border: 3px solid #2aa198; border-radius: 8px; box-shadow: 0 12px 30px rgba(0,0,0,0.35); }
                    .wrapper { padding: 24px; }
                </style>
            </head>
            <body>
                <div class="wrapper">
                    <h1>Live Face + Motion Detection</h1>
                    <p>Green boxes: faces. Blue boxes: motion. Refresh if stream stops.</p>
                    <img src="{{ url_for('stream') }}" alt="Camera stream">
                </div>
            </body>
            </html>
            """
        )

    @app.route("/stream")
    def stream():
        return Response(generate_frames(detector, capture), mimetype="multipart/x-mixed-replace; boundary=frame")

    return app


def generate_frames(detector: Detector, capture: cv2.VideoCapture):
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        annotated = detector.annotate(frame)
        ok, buffer = cv2.imencode(".jpg", annotated)
        if not ok:
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"


def main():
    args = parse_args()
    classifier = load_classifier()
    source = resolve_source(args.source)

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        sys.exit(f"Could not open video source: {source}")

    detector = Detector(
        classifier=classifier,
        scale=args.scale,
        neighbors=args.neighbors,
        min_size=args.min_size,
        min_motion_area=args.min_motion_area,
    )
    app = create_app(detector, capture)
    try:
        app.run(host="0.0.0.0", port=args.port, threaded=True)
    finally:
        capture.release()


if __name__ == "__main__":
    main()
