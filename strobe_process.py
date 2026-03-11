#!/usr/bin/env python3
"""
strobe_process.py

Offline strobe detection processor.
Reads a recorded .avi from ~/strobe_recordings/, runs strobe detection on
every frame, and writes an annotated output video.

Detection modes (--mode):
  delta    — original: bright AND brighter-than-background (default)
  simple   — absolute brightness threshold only, no background model
  flicker  — tracks brightness over time, confirms via FFT at target Hz

All modes return at most ONE detection per frame (the brightest candidate).

Run:
  python3 strobe_process.py                        # interactive picker, delta mode
  python3 strobe_process.py --mode flicker         # flicker mode
  python3 strobe_process.py --mode simple          # simple mode
  python3 strobe_process.py file.avi               # specific file
  python3 strobe_process.py --all                  # all unprocessed

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
from collections import deque

RECORDINGS_DIR  = os.path.expanduser("~/strobe_recordings")
OUTPUT_DIR      = os.path.expanduser("~/strobe_processed")
BOOT_OUTPUT_DIR = "/boot/strobe_output"


# ---------------------------------------------------------------------------
# Shared blob helpers
# ---------------------------------------------------------------------------

def _find_brightest_blob(gray, thresh, min_area):
    """
    Threshold gray, find contours, return the single brightest blob as a dict
    or None. 'Brightest' = highest mean pixel value inside the bounding rect.
    """
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_brightness = -1.0

    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        M = cv2.moments(cnt)
        if M["m00"] < 1e-6:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        bx, by, bw, bh = cv2.boundingRect(cnt)
        roi = gray[by:by + bh, bx:bx + bw]
        brightness = float(np.mean(roi)) if roi.size > 0 else 0.0

        if brightness > best_brightness:
            best_brightness = brightness
            best = {"cx": cx, "cy": cy, "brightness": brightness, "bbox": (bx, by, bw, bh)}

    return best


def _make_detection(blob, frame_w, frame_h, extra=None):
    cx, cy = blob["cx"], blob["cy"]
    dx = cx - frame_w / 2.0
    dy = frame_h / 2.0 - cy      # flip y: up = positive
    det = {
        "cx": cx, "cy": cy,
        "r":     math.hypot(dx, dy),
        "theta": math.degrees(math.atan2(dy, dx)),
        "brightness": blob["brightness"],
        "bbox": blob["bbox"],
    }
    if extra:
        det.update(extra)
    return det


# ---------------------------------------------------------------------------
# Mode 1 — Delta (original background-subtraction approach)
# ---------------------------------------------------------------------------

class DeltaDetector:
    """
    Detects pixels that are bright in absolute terms AND significantly brighter
    than the running background at that location.
    Assumes a mostly static camera.
    """
    def __init__(self, frame_width, frame_height,
                 min_blob_area=8, abs_brightness_thresh=200,
                 delta_thresh=70, background_alpha=0.025):
        self.w = frame_width
        self.h = frame_height
        self.min_blob_area = min_blob_area
        self.abs_brightness_thresh = abs_brightness_thresh
        self.delta_thresh = delta_thresh
        self.background_alpha = background_alpha
        self.bg = None

    def process(self, bgr_frame):
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

        best = None
        best_brightness = -1.0
        for cnt in contours:
            if cv2.contourArea(cnt) < self.min_blob_area:
                continue
            M = cv2.moments(cnt)
            if M["m00"] < 1e-6:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            bx, by, bw, bh = cv2.boundingRect(cnt)
            roi = gray[by:by + bh, bx:bx + bw]
            brightness = float(np.mean(roi)) if roi.size > 0 else 0.0
            if brightness > best_brightness:
                best_brightness = brightness
                best = {"cx": cx, "cy": cy, "brightness": brightness, "bbox": (bx, by, bw, bh)}

        if not best:
            cv2.accumulateWeighted(gray, self.bg, self.background_alpha)
            return []

        return [_make_detection(best, self.w, self.h)]


# ---------------------------------------------------------------------------
# Mode 2 — Simple (absolute brightness only, no background model)
# ---------------------------------------------------------------------------

class SimpleDetector:
    """
    No background model. Any blob above the absolute brightness threshold is
    a candidate. Returns only the single brightest one.
    Works regardless of camera motion.
    """
    def __init__(self, frame_width, frame_height,
                 min_blob_area=8, abs_brightness_thresh=200):
        self.w = frame_width
        self.h = frame_height
        self.min_blob_area = min_blob_area
        self.abs_brightness_thresh = abs_brightness_thresh

    def process(self, bgr_frame):
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        blob = _find_brightest_blob(gray, self.abs_brightness_thresh, self.min_blob_area)
        if blob is None:
            return []
        return [_make_detection(blob, self.w, self.h)]


# ---------------------------------------------------------------------------
# Mode 3 — Flicker (temporal FFT, motion-invariant)
# ---------------------------------------------------------------------------

class FlickerDetector:
    """
    Tracks the brightest bright-blob each frame and builds a brightness time
    series. Confirms as a strobe only if that signal has a dominant frequency
    in the target band (default 1–4 Hz centred at 2.5 Hz).

    Requires ~2 seconds of history before first confirmation, so the first
    ~60 frames will show no detection.
    """
    def __init__(self, frame_width, frame_height,
                 min_blob_area=8, abs_brightness_thresh=200,
                 target_hz=2.5, hz_tolerance=1.5,
                 history_s=3.0, min_history_s=2.0,
                 peak_ratio_thresh=0.25):
        self.w = frame_width
        self.h = frame_height
        self.min_blob_area = min_blob_area
        self.abs_brightness_thresh = abs_brightness_thresh
        self.target_hz = target_hz
        self.hz_tolerance = hz_tolerance
        self.history_s = history_s
        self.min_history_s = min_history_s
        self.peak_ratio_thresh = peak_ratio_thresh  # fraction of AC power in target band

        # Ring buffers — sized after first frame when fps is known
        self._fps = None
        self._brightness_history = deque()   # float, 0 when nothing detected
        self._blob_history = deque()          # blob dict or None

    def _init_buffers(self, fps):
        self._fps = fps
        max_len = int(fps * self.history_s)
        self._brightness_history = deque(maxlen=max_len)
        self._blob_history = deque(maxlen=max_len)

    def process(self, bgr_frame, fps=30.0):
        if self._fps is None:
            self._init_buffers(fps)

        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        blob = _find_brightest_blob(gray, self.abs_brightness_thresh, self.min_blob_area)
        self._brightness_history.append(blob["brightness"] if blob else 0.0)
        self._blob_history.append(blob)

        # Not enough history yet
        min_frames = int(self._fps * self.min_history_s)
        if len(self._brightness_history) < min_frames:
            return []

        signal = np.array(self._brightness_history, dtype=np.float32)

        # Reject if nothing has ever been bright in this window
        if signal.max() < self.abs_brightness_thresh * 0.7:
            return []

        # FFT on mean-subtracted signal (removes DC / constant lights)
        ac = signal - signal.mean()
        fft_mag = np.abs(np.fft.rfft(ac))
        freqs   = np.fft.rfftfreq(len(signal), d=1.0 / self._fps)

        lo = self.target_hz - self.hz_tolerance
        hi = self.target_hz + self.hz_tolerance
        in_band = (freqs >= lo) & (freqs <= hi)

        if not in_band.any():
            return []

        band_power  = fft_mag[in_band].max()
        total_power = fft_mag[1:].sum()   # exclude DC bin

        if total_power < 1e-6 or (band_power / total_power) < self.peak_ratio_thresh:
            return []

        detected_hz = float(freqs[in_band][np.argmax(fft_mag[in_band])])

        # Return the most recent actual blob position
        for blob in reversed(self._blob_history):
            if blob is not None:
                return [_make_detection(blob, self.w, self.h, extra={"freq_hz": detected_hz})]

        return []


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

BOX_COLOR    = (0,   0,   255)
TEXT_COLOR   = (0,   255, 255)
CENTRE_COLOR = (255, 255, 255)
FONT         = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE   = 0.45
THICKNESS    = 1


def annotate_frame(frame, detections, frame_w, frame_h):
    out = frame.copy()

    cx0, cy0 = frame_w // 2, frame_h // 2
    cv2.line(out, (cx0 - 12, cy0), (cx0 + 12, cy0), CENTRE_COLOR, 1)
    cv2.line(out, (cx0, cy0 - 12), (cx0, cy0 + 12), CENTRE_COLOR, 1)

    for det in detections:
        bx, by, bw, bh = det["bbox"]
        cv2.rectangle(out, (bx, by), (bx + bw, by + bh), BOX_COLOR, 2)

        label_lines = [
            f"r={det['r']:.1f}px  {det['theta']:.1f}deg",
            f"bright={det['brightness']:.0f}",
        ]
        if "freq_hz" in det:
            label_lines.append(f"flicker={det['freq_hz']:.2f}Hz")

        line_h = 16
        label_top = by - len(label_lines) * line_h - 4
        if label_top < 0:
            label_top = by + bh + 4

        for i, text in enumerate(label_lines):
            y = label_top + i * line_h
            cv2.putText(out, text, (bx + 1, y + 1), FONT, FONT_SCALE, (0, 0, 0), THICKNESS + 1, cv2.LINE_AA)
            cv2.putText(out, text, (bx,     y),     FONT, FONT_SCALE, TEXT_COLOR,  THICKNESS,     cv2.LINE_AA)

        cv2.line(out, (cx0, cy0), (det["cx"], det["cy"]), BOX_COLOR, 1, cv2.LINE_AA)

    return out


# ---------------------------------------------------------------------------
# File picker
# ---------------------------------------------------------------------------

def list_recordings():
    if not os.path.isdir(RECORDINGS_DIR):
        return []
    return sorted(
        f for f in os.listdir(RECORDINGS_DIR)
        if f.lower().endswith(".avi") and "_annotated" not in f
    )


def pick_files_interactive():
    files = list_recordings()
    if not files:
        print(f"No recordings found in {RECORDINGS_DIR}")
        return []

    print(f"\nRecordings in {RECORDINGS_DIR}:")
    for i, name in enumerate(files):
        path = os.path.join(RECORDINGS_DIR, name)
        size_mb = os.path.getsize(path) / 1e6
        done = os.path.exists(os.path.join(OUTPUT_DIR, name.replace(".avi", "_annotated.avi")))
        tag = "  [already processed]" if done else ""
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
    base = os.path.splitext(out_path)[0]
    for fourcc_str, ext in [("XVID", ".avi"), ("mp4v", ".mp4"), ("MJPG", ".avi")]:
        path = base + ext
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc_str), fps, (width, height))
        if writer.isOpened():
            print(f"  Codec: {fourcc_str} → {os.path.basename(path)}")
            return writer, path
        writer.release()
    return None, None


def build_detector(mode, width, height):
    if mode == "simple":
        return SimpleDetector(frame_width=width, frame_height=height)
    elif mode == "flicker":
        return FlickerDetector(frame_width=width, frame_height=height)
    else:
        return DeltaDetector(frame_width=width, frame_height=height)


def process_video(input_path, mode="delta"):
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
        print("ERROR: no working video codec found — install libxvidcore or libx264")
        cap.release()
        return

    detector = build_detector(mode, width, height)

    print(f"\nProcessing: {os.path.basename(input_path)}  [mode={mode}]")
    print(f"  {width}x{height}  {fps:.1f}fps  {total} frames → {os.path.basename(out_path)}")
    if mode == "flicker":
        print(f"  Note: first ~{int(fps * 2)} frames will show no detection (building history)")

    t_start      = time.time()
    frame_idx    = 0
    n_detections = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if mode == "flicker":
            detections = detector.process(frame, fps=fps)
        else:
            detections = detector.process(frame)

        n_detections += len(detections)
        writer.write(annotate_frame(frame, detections, width, height))

        frame_idx += 1
        if frame_idx % 100 == 0 or frame_idx == total:
            pct = 100 * frame_idx / total if total else 0
            elapsed = time.time() - t_start
            print(f"  {frame_idx}/{total}  ({pct:.0f}%)  {frame_idx/elapsed:.1f} fps processing")

    cap.release()
    writer.release()

    elapsed = time.time() - t_start
    size_mb = os.path.getsize(out_path) / 1e6 if os.path.exists(out_path) else 0
    print(f"  Done in {elapsed:.1f}s — {n_detections} detections — {size_mb:.1f} MB → {out_path}")

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
    parser.add_argument("files",   nargs="*", help="video file(s) to process")
    parser.add_argument("--all",   action="store_true", help="process all unprocessed recordings")
    parser.add_argument("--mode",  choices=["delta", "simple", "flicker"], default="delta",
                        help="detection mode (default: delta)")
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
        process_video(path, mode=args.mode)

    print("\nAll done.")


if __name__ == "__main__":
    main()
