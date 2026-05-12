#!/usr/bin/env python3
"""
breadboard_test.py

Breadboard peripheral test for Raspberry Pi 5.

Peripherals tested:
  - ADS1115 ADC   (I2C 0x48): reads A1, back-calculates 24V battery voltage
  - SSD1306 OLED  (I2C 0x3C, 128x64): displays live battery state
  - GPIO24 button (internal pull-up, active LOW): prints to terminal + OLED

Wiring:
  ADS1115 A1  -> voltage divider output (divides 24V battery to 20% = ~4.8V max)
  OLED SDA    -> Pi GPIO2  (I2C-1 SDA)
  OLED SCL    -> Pi GPIO3  (I2C-1 SCL)
  Button      -> GPIO24 to GND  (internal pull-up enabled)

Install (Pi 5, Bookworm):
  sudo apt install -y python3-pil python3-smbus python3-gpiozero i2c-tools python3-lgpio
  pip3 install adafruit-blinka \
      adafruit-circuitpython-ssd1306 \
      adafruit-circuitpython-ads1x15 \
      --break-system-packages
  sudo raspi-config nonint do_i2c 0
  sudo usermod -aG gpio $USER
  # reboot, then verify: i2cdetect -y 1  (expect 3c and 48)
"""

import time
import threading
import board
import busio
from PIL import Image, ImageDraw, ImageFont
from gpiozero import Button
import adafruit_ssd1306
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BUTTON_PIN      = 24
OLED_ADDR       = 0x3C
OLED_WIDTH      = 128
OLED_HEIGHT     = 64
ADS1115_ADDR    = 0x48
ADC_DIVIDER     = 0.20   # V_adc = V_battery * 0.20  =>  V_battery = V_adc / 0.20
ADC_GAIN        = 2 / 3  # ADS1115 PGA ±6.144 V — required for up to 4.8 V input
POLL_INTERVAL_S = 0.5
BTN_DISPLAY_S   = 1.0    # how long "PRESSED" stays on OLED after a press

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

_btn_event = threading.Event()

# ---------------------------------------------------------------------------
# Hardware init
# ---------------------------------------------------------------------------

button = Button(BUTTON_PIN, pull_up=True, bounce_time=0.05)
button.when_pressed = lambda: _btn_event.set()

i2c  = busio.I2C(board.SCL, board.SDA)
oled = adafruit_ssd1306.SSD1306_I2C(OLED_WIDTH, OLED_HEIGHT, i2c, addr=OLED_ADDR)
ads  = ADS.ADS1115(i2c, address=ADS1115_ADDR)
ads.gain = ADC_GAIN
chan = AnalogIn(ads, 1)  # channel A1

font = ImageFont.load_default()

# ---------------------------------------------------------------------------
# OLED
# ---------------------------------------------------------------------------

def draw_oled(v_batt: float, v_adc: float, btn_label: str) -> None:
    img  = Image.new("1", (OLED_WIDTH, OLED_HEIGHT))
    draw = ImageDraw.Draw(img)

    draw.text((0,  0), "=== BATTERY ===",               font=font, fill=255)
    draw.text((0, 16), f"  {v_batt:6.2f} V",            font=font, fill=255)
    draw.text((0, 32), f"ADC: {v_adc * 1000:6.1f} mV",  font=font, fill=255)
    draw.text((0, 48), f"BTN: {btn_label}",              font=font, fill=255)

    oled.image(img)
    oled.show()


def clear_oled() -> None:
    oled.fill(0)
    oled.show()

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    print("Breadboard test running. Ctrl-C to quit.")
    print(f"  OLED    : SSD1306 128x64 @ I2C 0x{OLED_ADDR:02X}")
    print(f"  ADC     : ADS1115 A1 @ I2C 0x{ADS1115_ADDR:02X}, gain=2/3 (±6.144 V), divider={ADC_DIVIDER}")
    print(f"  Button  : GPIO{BUTTON_PIN} pull-up (active LOW)\n")

    btn_pressed_at = 0.0

    try:
        while True:
            v_adc  = chan.voltage
            v_batt = v_adc / ADC_DIVIDER

            if _btn_event.is_set():
                _btn_event.clear()
                btn_pressed_at = time.time()
                print(f"[BTN] GPIO{BUTTON_PIN} pressed  |  V_adc={v_adc:.4f} V  V_batt={v_batt:.2f} V")

            btn_label = "PRESSED" if (time.time() - btn_pressed_at) < BTN_DISPLAY_S else "--"

            print(f"[ADC] V_adc={v_adc:.4f} V  V_batt={v_batt:.2f} V")
            draw_oled(v_batt, v_adc, btn_label)

            time.sleep(POLL_INTERVAL_S)

    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        clear_oled()
        button.close()


if __name__ == "__main__":
    main()
