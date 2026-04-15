#!/usr/bin/env python3
"""
button_listener.py

Waits for a button press on GPIO18 (BCM) / physical pin 12.
Button should be wired between GPIO18 and GND (internal pull-up enabled —
press pulls the pin LOW → falling edge triggers launch).

On press: launches Rpi_StrobeDetector.py with --no-led, and blinks GPIO15 LED
here while the recording runs. button_listener owns GPIO15 exclusively.
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
BUTTON_PIN  = 18   # BCM — physical pin 12
LED_PIN     = 15   # BCM — physical pin 22
BLINK_HZ    = 4.0
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
# LED
# ---------------------------------------------------------------------------

class ExternalLED:
    def __init__(self, pin: int):
        self.pin = pin
        GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)

    def set(self, on: bool):
        GPIO.output(self.pin, GPIO.HIGH if on else GPIO.LOW)

    def off(self):
        self.set(False)


class LEDBlinker(threading.Thread):
    """Blinks the LED at a fixed rate until stop() is called."""
    def __init__(self, led: ExternalLED, hz: float):
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

_lock           = threading.Lock()
led             = None   # set in main() after GPIO.setmode()
current_proc    = None
current_blinker = None


def _monitor(proc, blinker):
    """Wait for recording process to exit, then stop the blinker."""
    proc.wait()
    blinker.stop()
    blinker.join()
    log.info("Recording finished (PID %d) — LED off.", proc.pid)


def on_button_press(channel):
    global current_proc, current_blinker

    with _lock:
        if current_proc is not None and current_proc.poll() is None:
            log.info("Button pressed — recording already in progress, ignoring.")
            return

        log.info("Button pressed — launching %s", SCRIPT_PATH)

        try:
            current_proc = subprocess.Popen(
                [sys.executable, SCRIPT_PATH, "--no-led"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            log.info("Launched PID %d", current_proc.pid)

            # Blink LED here — strobe detector skips GPIO with --no-led
            current_blinker = LEDBlinker(led, BLINK_HZ)
            current_blinker.start()

            # Background thread stops the blinker when recording ends
            threading.Thread(
                target=_monitor,
                args=(current_proc, current_blinker),
                daemon=True,
            ).start()

            # Log child stderr without blocking
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

    log.info("Listening on GPIO%d (physical pin 12). Waiting for button press...", BUTTON_PIN)

    try:
        while True:
            channel = GPIO.wait_for_edge(BUTTON_PIN, GPIO.FALLING, timeout=1000)
            if channel is not None:
                on_button_press(channel)
    except KeyboardInterrupt:
        log.info("Shutting down.")
    finally:
        if current_blinker is not None:
            current_blinker.stop()
            current_blinker.join(timeout=1.0)
        led.off()
        GPIO.cleanup()


if __name__ == "__main__":
    main()
