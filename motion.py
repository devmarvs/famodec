import argparse
import sys
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time face and motion detection with OpenCV.")
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index (e.g. 0) or path to a video file. Defaults to the first camera.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.2,
        help="How much the image size is reduced at each image scale.",
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
    motion_group = parser.add_mutually_exclusive_group()
    motion_group.add_argument(
        "--motion",
        dest="motion",
        action="store_true",
        help="Enable motion detection overlay (default).",
    )
    motion_group.add_argument(
        "--no-motion",
        dest="motion",
        action="store_false",
        help="Disable motion detection overlay.",
    )
    parser.set_defaults(motion=True)
    parser.add_argument(
        "--min-motion-area",
        type=int,
        default=500,
        help="Minimum contour area to treat as motion (in pixels).",
    )
    return parser.parse_args()


def load_classifier() -> cv2.CascadeClassifier:
    """Load the Haar cascade shipped with OpenCV."""
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    classifier = cv2.CascadeClassifier(str(cascade_path))
    if classifier.empty():
        raise RuntimeError(f"Failed to load cascade from {cascade_path}")
    return classifier


def resolve_source(source: str):
    """Convert numeric camera index strings to ints for VideoCapture."""
    if source.isdigit():
        return int(source)
    return source


def main() -> None:
    args = parse_args()
    classifier = load_classifier()
    source = resolve_source(args.source)

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        sys.exit(f"Could not open video source: {source}")

    window_name = "Face + Motion Detection"
    prev_gray = None  # For motion detection frame differencing.
    while True:
        ok, frame = capture.read()
        if not ok:
            print("Could not read frame, stopping.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_boxes = []
        if args.motion:
            blurred = cv2.GaussianBlur(gray, (21, 21), 0)
            if prev_gray is None:
                prev_gray = blurred
            delta = cv2.absdiff(prev_gray, blurred)
            thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < args.min_motion_area:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                motion_boxes.append((x, y, w, h))
            prev_gray = blurred

        faces = classifier.detectMultiScale(
            gray,
            scaleFactor=args.scale,
            minNeighbors=args.neighbors,
            minSize=(args.min_size, args.min_size),
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
            f"Faces: {len(faces)} | Motion: {len(motion_boxes)} | Press q/Esc to quit",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
