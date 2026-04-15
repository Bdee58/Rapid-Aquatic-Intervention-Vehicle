#!/usr/bin/env python3
"""
Rpi_StrobeDetector.py

Raspberry Pi Zero 2 W + Camera Module 3
Records video to ~/strobe_recordings/ with a timestamped filename.
Stop recording with Ctrl+C.

Requires:
  sudo apt update
  sudo apt install -y python3-picamera2 python3-opencv python3-numpy

Run:
  python3 Rpi_StrobeDetector.py
  python3 Rpi_StrobeDetector.py --width 640 --height 480 --fps 30
"""

import os
import sys
import cv2
import time
import argparse
import threading
import numpy as np
from datetime import datetime
from picamera2 import Picamera2
import RPi.GPIO as GPIO

RECORDINGS_DIR = os.path.expanduser("~/strobe_recordings")
LED_PIN = 15  # BCM — physical pin 22


class LEDBlinker(threading.Thread):
    """Blinks an output GPIO pin at a fixed rate until stop() is called."""
    def __init__(self, pin: int, hz: float = 4.0):
        super().__init__(daemon=True)
        self._pin  = pin
        self._half = max(0.05, 0.5 / hz)
        self._stop = threading.Event()

    def run(self):
        state = False
        while not self._stop.is_set():
            state = not state
            GPIO.output(self._pin, GPIO.HIGH if state else GPIO.LOW)
            time.sleep(self._half)
        GPIO.output(self._pin, GPIO.LOW)  # ensure LED off on exit

    def stop(self):
        self._stop.set()


def build_camera(width, height, fps):
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (width, height), "format": "RGB888"},
        buffer_count=4,
    )
    picam2.configure(config)
    picam2.set_controls({
        "FrameRate": fps,
        # Lock exposure/gain for consistent strobe detection during processing.
        # Tune these values for your pool environment.
        # "ExposureTime": 8000,   # microseconds — uncomment and tune as needed
        # "AnalogueGain": 4.0,    # uncomment and tune as needed
    })
    picam2.start()
    time.sleep(1.0)  # let camera settle
    return picam2


def main():
    parser = argparse.ArgumentParser(description="Record strobe detection video on Raspberry Pi")
    parser.add_argument("--width",    type=int,   default=320)
    parser.add_argument("--height",   type=int,   default=240)
    parser.add_argument("--fps",      type=int,   default=30)
    parser.add_argument("--duration", type=float, default=30.0, help="recording duration in seconds")
    args = parser.parse_args()

    os.makedirs(RECORDINGS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = os.path.join(RECORDINGS_DIR, f"strobe_{timestamp}.avi")

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(out_path, fourcc, args.fps, (args.width, args.height))
    if not writer.isOpened():
        print(f"ERROR: could not open VideoWriter for {out_path}", file=sys.stderr)
        sys.exit(1)

    picam2 = build_camera(args.width, args.height, args.fps)

    # --- LED setup ---
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(LED_PIN, GPIO.OUT, initial=GPIO.LOW)
    blinker = LEDBlinker(LED_PIN, hz=4.0)
    blinker.start()

    print(f"Recording {args.duration:.0f}s → {out_path}  (LED on GPIO{LED_PIN} blinking)")

    frame_count = 0
    t_start = time.time()

    try:
        while True:
            elapsed = time.time() - t_start
            if elapsed >= args.duration:
                break

            frame_rgb = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
            frame_count += 1

            if frame_count % (args.fps * 5) == 0:
                remaining = args.duration - elapsed
                print(f"  {elapsed:.0f}s elapsed — {remaining:.0f}s remaining — {frame_count} frames")

    except KeyboardInterrupt:
        print("\nStopped early by user.")
    finally:
        blinker.stop()
        blinker.join(timeout=1.0)
        GPIO.cleanup()
        picam2.stop()
        writer.release()
        elapsed = time.time() - t_start
        size_mb = os.path.getsize(out_path) / 1e6 if os.path.exists(out_path) else 0
        print(f"\nSaved {frame_count} frames ({elapsed:.1f}s) → {out_path}  [{size_mb:.1f} MB]")


if __name__ == "__main__":
    main()
