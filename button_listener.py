#!/usr/bin/env python3
"""
button_listener.py

Waits for a button press on GPIO14 (BCM) / physical pin 8.
Button should be wired between GPIO14 and GND (internal pull-up enabled —
press pulls the pin LOW → falling edge triggers launch).

On press: flashes the LED 3× to confirm, then launches Rpi_StrobeDetector.py.
LED blinking during recording is handled by Rpi_StrobeDetector.py itself.
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
BUTTON_PIN  = 14   # BCM — physical pin 8
LED_PIN     = 15   # BCM — physical pin 22
DEBOUNCE_MS = 300
SCRIPT_PATH = os.path.expanduser(
    "~/Rapid-Aquatic-Intervention-Vehicle/Rpi_StrobeDetector.py"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [button_listener] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LED helpers
# ---------------------------------------------------------------------------

class ExternalLED:
    """Controls an external LED wired to a GPIO output pin."""
    def __init__(self, pin: int):
        self.pin = pin
        GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)

    def set(self, on: bool):
        GPIO.output(self.pin, GPIO.HIGH if on else GPIO.LOW)

    def off(self):
        self.set(False)


def confirmation_flash(led: ExternalLED, n: int = 3) -> None:
    """Flash LED n times quickly to acknowledge the button press.

    Runs in a short daemon thread so the interrupt callback returns immediately.
    Rpi_StrobeDetector.py will take over the pin once it starts.
    """
    def _flash():
        for _ in range(n):
            led.set(True)
            time.sleep(0.10)
            led.set(False)
            time.sleep(0.10)
    threading.Thread(target=_flash, daemon=True).start()


# ---------------------------------------------------------------------------
# Button handler
# ---------------------------------------------------------------------------

_lock        = threading.Lock()   # guards current_proc
led          = None               # set in main() after GPIO.setmode()
current_proc = None


def on_button_press(channel):
    global current_proc

    with _lock:
        # Ignore press if a recording is still running
        if current_proc is not None and current_proc.poll() is None:
            log.info("Button pressed — recording already in progress, ignoring.")
            return

        log.info("Button pressed — launching %s", SCRIPT_PATH)

        if led is not None:
            confirmation_flash(led)

        try:
            current_proc = subprocess.Popen(
                [sys.executable, SCRIPT_PATH],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,   # capture stderr for logging
            )
            log.info("Launched PID %d", current_proc.pid)

            # Log any startup errors from the child without blocking
            def _log_stderr(proc):
                for line in proc.stderr:
                    log.warning("[strobe] %s", line.decode(errors="replace").rstrip())
            threading.Thread(target=_log_stderr, args=(current_proc,), daemon=True).start()

        except Exception as e:
            log.error("Failed to launch script: %s", e)


def main():
    global led

    if not os.path.exists(SCRIPT_PATH):
        log.error("Script not found: %s", SCRIPT_PATH)
        sys.exit(1)

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    led = ExternalLED(LED_PIN)

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
        led.off()      # ensure LED is low before releasing the pin
        GPIO.cleanup()


if __name__ == "__main__":
    main()
