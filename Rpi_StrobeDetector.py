#!/usr/bin/env python3
"""
strobe_home_led.py

Raspberry Pi Zero 2 W + Camera Module 3
Detect a flashing strobe and blink the onboard LED at a rate proportional
to how centered the strobe is in the camera frame.

Tested design assumptions:
- Dark scene / pool environment
- Strobe is the brightest transient object in view
- You care about horizontal centering most for homing
- LED blink frequency increases as centering improves

Requires:
  sudo apt update
  sudo apt install -y python3-picamera2 python3-opencv python3-numpy

Run:
  sudo python3 strobe_home_led.py
or:
  sudo python3 strobe_home_led.py --debug
"""

import os
import sys
import cv2
import time
import math
import argparse
import threading
import numpy as np

from picamera2 import Picamera2


# ---------------------------
# LED control
# ---------------------------

class OnboardLED:
    """
    Controls the Pi ACT LED via /sys/class/leds.
    On many Raspberry Pi OS installs, the LED appears as /sys/class/leds/led0
    or /sys/class/leds/ACT, with trigger/brightness files. Paths and polarity
    can vary by model/OS, so this class probes common cases. :contentReference[oaicite:1]{index=1}
    """
    def __init__(self):
        self.led_path = self._find_led_path()
        self.brightness_path = None
        self.trigger_path = None
        self.inverted = False

        if self.led_path:
            self.brightness_path = os.path.join(self.led_path, "brightness")
            self.trigger_path = os.path.join(self.led_path, "trigger")
            self._set_trigger_none()

    def _find_led_path(self):
        candidates = [
            "/sys/class/leds/led0",
            "/sys/class/leds/ACT",
            "/sys/class/leds/act",
        ]
        for c in candidates:
            if os.path.isdir(c):
                return c
        leds_root = "/sys/class/leds"
        if os.path.isdir(leds_root):
            for name in os.listdir(leds_root):
                p = os.path.join(leds_root, name)
                if os.path.isdir(p) and os.path.exists(os.path.join(p, "brightness")):
                    return p
        return None

    def _write(self, path, value):
        with open(path, "w") as f:
            f.write(str(value))

    def _read(self, path):
        with open(path, "r") as f:
            return f.read().strip()

    def _set_trigger_none(self):
        if self.trigger_path and os.path.exists(self.trigger_path):
            try:
                self._write(self.trigger_path, "none")
            except PermissionError:
                print("Need sudo/root to control onboard LED.", file=sys.stderr)
                raise

    def set(self, on: bool):
        if not self.brightness_path:
            return
        val = 1 if on else 0
        # Most Pi models use non-inverted logic for led0 brightness,
        # though some newer models can differ. This fallback keeps code simple.
        try:
            self._write(self.brightness_path, str(val))
        except PermissionError:
            print("Need sudo/root to control onboard LED.", file=sys.stderr)
            raise

    def off(self):
        self.set(False)


class LEDBlinker(threading.Thread):
    """
    Separate thread so LED blinking is independent of camera loop timing.
    """
    def __init__(self, led: OnboardLED):
        super().__init__(daemon=True)
        self.led = led
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

        # Hz
        self.target_hz = 0.5
        self.enabled = True

    def update_rate(self, hz: float, enabled: bool = True):
        with self.lock:
            self.target_hz = max(0.0, float(hz))
            self.enabled = enabled

    def run(self):
        state = False
        while not self.stop_event.is_set():
            with self.lock:
                hz = self.target_hz
                enabled = self.enabled

            if not enabled or hz <= 0.0:
                if state:
                    self.led.off()
                    state = False
                time.sleep(0.05)
                continue

            half_period = max(0.03, 0.5 / hz)
            state = not state
            self.led.set(state)
            time.sleep(half_period)

        self.led.off()

    def stop(self):
        self.stop_event.set()


# ---------------------------
# Strobe detector
# ---------------------------

class StrobeTracker:
    def __init__(
        self,
        frame_width=320,
        frame_height=240,
        min_blob_area=6,
        abs_brightness_thresh=180,
        delta_thresh=45,
        background_alpha=0.03,
        lock_timeout_s=0.4,
    ):
        self.w = frame_width
        self.h = frame_height
        self.min_blob_area = min_blob_area
        self.abs_brightness_thresh = abs_brightness_thresh
        self.delta_thresh = delta_thresh
        self.background_alpha = background_alpha
        self.lock_timeout_s = lock_timeout_s

        self.bg = None
        self.last_detection_time = 0.0
        self.last_center = None
        self.last_area = 0.0

    def process(self, bgr_frame):
        """
        Returns:
            found: bool
            cx, cy: blob center if found else None
            score: 0..1 centeredness score
            overlay: debug visualization
        """
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.bg is None:
            self.bg = gray.astype(np.float32)
            overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            return False, None, None, 0.0, overlay

        # Positive transients relative to running background
        bg_u8 = cv2.convertScaleAbs(self.bg)
        delta = cv2.subtract(gray, bg_u8)

        # Bright and changing
        _, m1 = cv2.threshold(gray, self.abs_brightness_thresh, 255, cv2.THRESH_BINARY)
        _, m2 = cv2.threshold(delta, self.delta_thresh, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(m1, m2)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        found = False
        cx = cy = None
        area = 0.0

        if contours:
            best = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(best)
            if area >= self.min_blob_area:
                M = cv2.moments(best)
                if M["m00"] > 1e-6:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    found = True
                    self.last_detection_time = time.time()
                    self.last_center = (cx, cy)
                    self.last_area = area

        # Update background slowly so flashes do not get absorbed too fast
        cv2.accumulateWeighted(gray, self.bg, self.background_alpha)

        locked = (time.time() - self.last_detection_time) <= self.lock_timeout_s

        if found:
            dx = abs(cx - self.w / 2.0)
            score = max(0.0, 1.0 - dx / (self.w / 2.0))
        elif locked and self.last_center is not None:
            cx, cy = self.last_center
            dx = abs(cx - self.w / 2.0)
            score = max(0.0, 1.0 - dx / (self.w / 2.0))
        else:
            score = 0.0

        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.line(overlay, (self.w // 2, 0), (self.w // 2, self.h - 1), (255, 0, 0), 1)

        if found and cx is not None:
            cv2.circle(overlay, (cx, cy), 8, (0, 255, 0), 2)
            cv2.putText(
                overlay,
                f"FOUND x={cx} score={score:.2f} area={area:.0f}",
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        elif locked and self.last_center is not None:
            cv2.circle(overlay, self.last_center, 8, (0, 255, 255), 2)
            cv2.putText(
                overlay,
                f"RECENT LOCK score={score:.2f}",
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                overlay,
                "NO LOCK",
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        return found, cx, cy, score, overlay


# ---------------------------
# Main
# ---------------------------

def map_score_to_blink_hz(score, has_lock,
                          no_lock_hz=0.5,
                          edge_hz=1.5,
                          centered_hz=8.0):
    """
    score = 0 at far edge, 1 at centered.
    """
    if not has_lock:
        return no_lock_hz
    return edge_hz + score * (centered_hz - edge_hz)


def build_camera(width=320, height=240, fps=30):
    picam2 = Picamera2()

    # Picamera2 supports low-res processing streams suitable for live analysis. :contentReference[oaicite:2]{index=2}
    config = picam2.create_preview_configuration(
        main={"size": (width, height), "format": "RGB888"},
        buffer_count=4
    )
    picam2.configure(config)
    picam2.set_controls({"FrameRate": fps})
    picam2.start()
    time.sleep(1.0)
    return picam2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="show debug window")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    led = OnboardLED()
    blinker = LEDBlinker(led)
    blinker.start()

    tracker = StrobeTracker(
        frame_width=args.width,
        frame_height=args.height,
        min_blob_area=8,
        abs_brightness_thresh=185,
        delta_thresh=40,
        background_alpha=0.025,
        lock_timeout_s=0.35,
    )

    picam2 = build_camera(args.width, args.height, args.fps)

    try:
        while True:
            frame = picam2.capture_array()

            # Picamera2 RGB888 comes in RGB order; OpenCV expects BGR for many ops.
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            found, cx, cy, score, overlay = tracker.process(frame_bgr)

            has_lock = found or ((time.time() - tracker.last_detection_time) <= tracker.lock_timeout_s)
            blink_hz = map_score_to_blink_hz(score, has_lock)

            blinker.update_rate(blink_hz, enabled=True)

            if args.debug:
                cv2.putText(
                    overlay,
                    f"blink={blink_hz:.2f} Hz",
                    (8, 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow("strobe_tracker", overlay)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    break

    finally:
        blinker.stop()
        blinker.join(timeout=1.0)
        picam2.stop()
        if args.debug:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()