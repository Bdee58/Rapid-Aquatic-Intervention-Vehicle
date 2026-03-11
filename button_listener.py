#!/usr/bin/env python3
"""
button_listener.py

Waits for a button press on GPIO14 (BCM) / physical pin 8.
Button should be wired between GPIO14 and GND with internal pull-up enabled
(press pulls the pin LOW → falling edge triggers launch).

When pressed, launches Rpi_StrobeDetector.py.
Ignores presses while a recording is already running.

Runs on boot via systemd — see button-listener.service.
"""

import os
import sys
import time
import logging
import threading
import subprocess
import RPi.GPIO as GPIO

# --- Config ---
BUTTON_PIN   = 14   # BCM numbering — change to 17 if that's your actual wiring
DEBOUNCE_MS  = 300
BLINK_HZ     = 4.0  # LED blink rate while recording
SCRIPT_PATH  = os.path.expanduser(
    "~/Rapid-Aquatic-Intervention-Vehicle/Rpi_StrobeDetector.py"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [button_listener] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LED
# ---------------------------------------------------------------------------

class OnboardLED:
    """Controls the Pi ACT LED via /sys/class/leds."""
    def __init__(self):
        self.led_path = self._find_led_path()
        self.brightness_path = os.path.join(self.led_path, "brightness") if self.led_path else None
        self.trigger_path    = os.path.join(self.led_path, "trigger")    if self.led_path else None
        if self.trigger_path and os.path.exists(self.trigger_path):
            self._write(self.trigger_path, "none")

    def _find_led_path(self):
        for c in ["/sys/class/leds/led0", "/sys/class/leds/ACT", "/sys/class/leds/act"]:
            if os.path.isdir(c):
                return c
        root = "/sys/class/leds"
        if os.path.isdir(root):
            for name in os.listdir(root):
                p = os.path.join(root, name)
                if os.path.isdir(p) and os.path.exists(os.path.join(p, "brightness")):
                    return p
        return None

    def _write(self, path, value):
        try:
            with open(path, "w") as f:
                f.write(str(value))
        except Exception as e:
            log.warning("LED write failed: %s", e)

    def set(self, on: bool):
        if self.brightness_path:
            self._write(self.brightness_path, "1" if on else "0")

    def off(self):
        self.set(False)


class LEDBlinker(threading.Thread):
    """Blinks the LED at a fixed rate until stop() is called."""
    def __init__(self, led: OnboardLED, hz: float):
        super().__init__(daemon=True)
        self._led  = led
        self._hz   = hz
        self._stop = threading.Event()

    def run(self):
        half = max(0.05, 0.5 / self._hz)
        state = False
        while not self._stop.is_set():
            state = not state
            self._led.set(state)
            time.sleep(half)
        self._led.off()

    def stop(self):
        self._stop.set()


# ---------------------------------------------------------------------------
# Button handler
# ---------------------------------------------------------------------------

led     = OnboardLED()
current_proc    = None
current_blinker = None


def _monitor(proc, blinker):
    """Background thread: waits for recording to finish, then kills the blinker."""
    proc.wait()
    blinker.stop()
    blinker.join()
    log.info("Recording finished (PID %d) — LED off.", proc.pid)


def on_button_press(channel):
    global current_proc, current_blinker

    # If a recording is still running, ignore the press
    if current_proc is not None and current_proc.poll() is None:
        log.info("Button pressed — recording already in progress, ignoring.")
        return

    log.info("Button pressed — launching %s", SCRIPT_PATH)
    try:
        current_proc = subprocess.Popen(
            [sys.executable, SCRIPT_PATH],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        log.info("Launched PID %d", current_proc.pid)

        current_blinker = LEDBlinker(led, BLINK_HZ)
        current_blinker.start()

        threading.Thread(
            target=_monitor,
            args=(current_proc, current_blinker),
            daemon=True,
        ).start()

    except Exception as e:
        log.error("Failed to launch script: %s", e)


def main():
    if not os.path.exists(SCRIPT_PATH):
        log.error("Script not found: %s", SCRIPT_PATH)
        sys.exit(1)

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.add_event_detect(
        BUTTON_PIN,
        GPIO.FALLING,
        callback=on_button_press,
        bouncetime=DEBOUNCE_MS,
    )

    log.info("Listening on GPIO%d (physical pin 8). Waiting for button press...", BUTTON_PIN)

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        log.info("Shutting down.")
    finally:
        GPIO.cleanup()


if __name__ == "__main__":
    main()
