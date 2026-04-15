#!/usr/bin/env python3
"""
blink_on_button.py
Blinks the LED on GPIO15 when the button on GPIO14 is pressed.
Button: GPIO14 (pulled up internally, press connects to GND)
LED:    GPIO15 (active high, external current-limiting resistor required)
"""

import RPi.GPIO as GPIO
import time

BUTTON_PIN = 18  # GPIO14/15 are UART TX/RX — edge detection fails on them
LED_PIN    = 15

BLINK_ON_S  = 0.3   # seconds LED stays on
BLINK_OFF_S = 0.3   # seconds LED stays off
BLINK_COUNT = 5     # number of blinks per button press

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(LED_PIN, GPIO.OUT, initial=GPIO.LOW)

print("Ready — press the button (GPIO14) to blink the LED (GPIO15). Ctrl-C to quit.")

try:
    while True:
        # Wait for falling edge (button press pulls line LOW)
        GPIO.wait_for_edge(BUTTON_PIN, GPIO.FALLING)
        time.sleep(0.05)  # debounce

        for _ in range(BLINK_COUNT):
            GPIO.output(LED_PIN, GPIO.HIGH)
            time.sleep(BLINK_ON_S)
            GPIO.output(LED_PIN, GPIO.LOW)
            time.sleep(BLINK_OFF_S)

except KeyboardInterrupt:
    pass
finally:
    GPIO.output(LED_PIN, GPIO.LOW)
    GPIO.cleanup()
    print("GPIO cleaned up.")
