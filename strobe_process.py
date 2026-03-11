#!/usr/bin/env python3
"""
strobe_process.py

Offline strobe detection processor.
Reads a recorded .avi from ~/strobe_recordings/, runs strobe detection on
every frame, and writes an annotated output video alongside the original.

Run:
  python3 strobe_process.py            # interactive file picker
  python3 strobe_process.py file.avi   # process specific file(s) directly
  python3 strobe_process.py --all      # process all unprocessed recordings

Output is saved as  <original_name>_annotated.avi  in the same directory.

Requires:
  sudo apt install -y python3-opencv python3-numpy
"""

import os
import sys
import cv2
import math
import time
import argparse
import numpy as np

RECORDINGS_DIR = os.path.expanduser("~/strobe_recordings")

# Annotated outputs are written here first (ext4, reliable), then copied to
# the boot partition so they're visible when the SD card is plugged into a laptop.
# Bullseye: /boot/strobe_output   Bookworm: /boot/firmware/strobe_output
OUTPUT_DIR      = os.path.expanduser("~/strobe_processed")
BOOT_OUTPUT_DIR = "/boot/strobe_output"


# ---------------------------------------------------------------------------
# Strobe detection
# ---------------------------------------------------------------------------

class StrobeDetector:
    def __init__(
        self,
        frame_width,
        frame_height,
        min_blob_area=8,
        abs_brightness_thresh=185,
        delta_thresh=40,
        background_alpha=0.025,
    ):
        self.w = frame_width
        self.h = frame_height
        self.min_blob_area = min_blob_area
        self.abs_brightness_thresh = abs_brightness_thresh
        self.delta_thresh = delta_thresh
        self.background_alpha = background_alpha
        self.bg = None

    def process(self, bgr_frame):
        """
        Returns a list of detections (may be empty). Each detection is a dict:
            cx, cy      : blob centroid (pixels)
            r, theta    : polar coords relative to frame centre (px, degrees)
            brightness  : mean pixel value inside bounding rect
            bbox        : (x, y, w, h) bounding rect
        """
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.bg is None:
            self.bg = gray.astype(np.float32)
            return []

        bg_u8 = cv2.convertScaleAbs(self.bg)
        delta = cv2.subtract(gray, bg_u8)

        _, m1 = cv2.threshold(gray,  self.abs_brightness_thresh, 255, cv2.THRESH_BINARY)
        _, m2 = cv2.threshold(delta, self.delta_thresh,           255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(m1, m2)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_blob_area:
                continue

            M = cv2.moments(cnt)
            if M["m00"] < 1e-6:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Polar coords relative to frame centre (right = 0°, CCW positive)
            dx = cx - self.w / 2.0
            dy = self.h / 2.0 - cy          # flip y so up is positive
            r     = math.hypot(dx, dy)
            theta = math.degrees(math.atan2(dy, dx))

            # Brightness: mean inside bounding rect
            bx, by, bw, bh = cv2.boundingRect(cnt)
            roi = gray[by:by + bh, bx:bx + bw]
            brightness = float(np.mean(roi)) if roi.size > 0 else 0.0

            detections.append({
                "cx": cx, "cy": cy,
                "r": r, "theta": theta,
                "brightness": brightness,
                "bbox": (bx, by, bw, bh),
                "area": area,
            })

        # Only update background on frames with no detections so strobes don't
        # get absorbed into the background model.
        if not detections:
            cv2.accumulateWeighted(gray, self.bg, self.background_alpha)

        return detections


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

# Colour scheme
BOX_COLOR    = (0,   0,   255)   # red
TEXT_COLOR   = (0,   255, 255)   # yellow
CENTRE_COLOR = (255, 255, 255)   # white

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45
THICKNESS  = 1


def annotate_frame(frame, detections, frame_w, frame_h):
    out = frame.copy()

    # Frame-centre crosshair
    cx0, cy0 = frame_w // 2, frame_h // 2
    cv2.line(out, (cx0 - 12, cy0), (cx0 + 12, cy0), CENTRE_COLOR, 1)
    cv2.line(out, (cx0, cy0 - 12), (cx0, cy0 + 12), CENTRE_COLOR, 1)

    for det in detections:
        bx, by, bw, bh = det["bbox"]

        # Bounding box
        cv2.rectangle(out, (bx, by), (bx + bw, by + bh), BOX_COLOR, 2)

        # Label lines — stacked above the box, or below if near top edge
        label_lines = [
            f"r={det['r']:.1f}px  {det['theta']:.1f}deg",
            f"bright={det['brightness']:.0f}",
        ]

        line_h = 16
        label_top = by - len(label_lines) * line_h - 4
        if label_top < 0:
            label_top = by + bh + 4   # flip below box

        for i, text in enumerate(label_lines):
            y = label_top + i * line_h
            # Shadow for readability
            cv2.putText(out, text, (bx + 1, y + 1), FONT, FONT_SCALE, (0, 0, 0), THICKNESS + 1, cv2.LINE_AA)
            cv2.putText(out, text, (bx,     y),     FONT, FONT_SCALE, TEXT_COLOR,  THICKNESS,     cv2.LINE_AA)

        # Line from centre to blob
        cv2.line(out, (cx0, cy0), (det["cx"], det["cy"]), BOX_COLOR, 1, cv2.LINE_AA)

    return out


# ---------------------------------------------------------------------------
# File picker
# ---------------------------------------------------------------------------

def list_recordings():
    if not os.path.isdir(RECORDINGS_DIR):
        return []
    files = sorted(
        f for f in os.listdir(RECORDINGS_DIR)
        if f.lower().endswith(".avi") and "_annotated" not in f
    )
    return files


def pick_files_interactive():
    files = list_recordings()
    if not files:
        print(f"No recordings found in {RECORDINGS_DIR}")
        return []

    print(f"\nRecordings in {RECORDINGS_DIR}:")
    for i, name in enumerate(files):
        path = os.path.join(RECORDINGS_DIR, name)
        size_mb = os.path.getsize(path) / 1e6
        annotated = os.path.exists(os.path.join(OUTPUT_DIR, name.replace(".avi", "_annotated.avi")))
        tag = "  [already processed]" if annotated else ""
        print(f"  [{i + 1}] {name}  ({size_mb:.1f} MB){tag}")

    print("\nEnter numbers to process (space-separated), or 'all': ", end="", flush=True)
    raw = input().strip().lower()

    if raw == "all":
        return [os.path.join(RECORDINGS_DIR, f) for f in files]

    selected = []
    for tok in raw.split():
        try:
            idx = int(tok) - 1
            if 0 <= idx < len(files):
                selected.append(os.path.join(RECORDINGS_DIR, files[idx]))
            else:
                print(f"  Ignoring out-of-range index: {tok}")
        except ValueError:
            print(f"  Ignoring invalid input: {tok}")

    return selected


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def open_writer(out_path, fps, width, height):
    """Try codecs in order, return (writer, actual_path) or (None, None)."""
    candidates = [
        ("XVID", ".avi"),
        ("mp4v", ".mp4"),
        ("MJPG", ".avi"),
    ]
    base = os.path.splitext(out_path)[0]
    for fourcc_str, ext in candidates:
        path = base + ext
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc_str), fps, (width, height))
        if writer.isOpened():
            print(f"  Codec: {fourcc_str} → {os.path.basename(path)}")
            return writer, path
        writer.release()
    return None, None


def process_video(input_path):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    basename = os.path.splitext(os.path.basename(input_path))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{basename}_annotated.avi")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"ERROR: cannot open {input_path}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer, out_path = open_writer(out_path, fps, width, height)
    if writer is None:
        print(f"ERROR: no working video codec found — install libxvidcore or libx264")
        cap.release()
        return

    detector = StrobeDetector(frame_width=width, frame_height=height)

    print(f"\nProcessing: {os.path.basename(input_path)}")
    print(f"  {width}x{height}  {fps:.1f}fps  {total} frames → {os.path.basename(out_path)}")

    t_start   = time.time()
    frame_idx = 0
    n_detections = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.process(frame)
        n_detections += len(detections)
        annotated = annotate_frame(frame, detections, width, height)
        writer.write(annotated)

        frame_idx += 1
        if frame_idx % 100 == 0 or frame_idx == total:
            pct = 100 * frame_idx / total if total else 0
            elapsed = time.time() - t_start
            fps_actual = frame_idx / elapsed if elapsed > 0 else 0
            print(f"  {frame_idx}/{total}  ({pct:.0f}%)  {fps_actual:.1f} fps processing")

    cap.release()
    writer.release()

    elapsed = time.time() - t_start
    size_mb = os.path.getsize(out_path) / 1e6 if os.path.exists(out_path) else 0
    print(f"  Done in {elapsed:.1f}s — {n_detections} strobe detections — saved {size_mb:.1f} MB → {out_path}")

    # Copy to boot partition so it's visible when SD card is plugged into a laptop
    try:
        import shutil
        os.makedirs(BOOT_OUTPUT_DIR, exist_ok=True)
        boot_dest = os.path.join(BOOT_OUTPUT_DIR, os.path.basename(out_path))
        shutil.copy2(out_path, boot_dest)
        print(f"  Copied to boot → {boot_dest}")
    except Exception as e:
        print(f"  Warning: could not copy to boot partition ({e})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Offline strobe detection annotator")
    parser.add_argument("files",  nargs="*", help="video file(s) to process (default: interactive picker)")
    parser.add_argument("--all",  action="store_true", help="process all unprocessed recordings non-interactively")
    args = parser.parse_args()

    if args.all:
        targets = [
            os.path.join(RECORDINGS_DIR, f)
            for f in list_recordings()
            if not os.path.exists(os.path.join(OUTPUT_DIR, f.replace(".avi", "_annotated.avi")))
        ]
        if not targets:
            print("Nothing to process.")
            return
    elif args.files:
        targets = []
        for f in args.files:
            # Accept bare filename (assumed in RECORDINGS_DIR) or full path
            if os.path.isabs(f) or os.path.exists(f):
                targets.append(f)
            else:
                targets.append(os.path.join(RECORDINGS_DIR, f))
    else:
        targets = pick_files_interactive()

    if not targets:
        print("No files selected. Exiting.")
        return

    for path in targets:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
        process_video(path)

    print("\nAll done.")


if __name__ == "__main__":
    main()
