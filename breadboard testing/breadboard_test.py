#!/usr/bin/env python3
"""
breadboard_test.py

Breadboard peripheral test for Raspberry Pi 5.

Peripherals tested:
  - ADS1115 ADC   (I2C 0x48): reads A1, back-calculates 24V battery voltage
  - SSD1306 OLED  (I2C 0x3C, 128x64): displays battery state + IMU orientation
  - MPU-6050 IMU  (I2C 0x68): pitch, roll (accel), yaw (gyro integration — drifts)
  - GPIO24 button (internal pull-up, active LOW): prints to terminal + OLED

Wiring:
  ADS1115 A1  -> voltage divider output (divides 24V battery to 20% = ~4.8V max)
  OLED SDA    -> Pi GPIO2  (I2C-1 SDA)
  OLED SCL    -> Pi GPIO3  (I2C-1 SCL)
  MPU-6050    -> same I2C bus, addr 0x68, powered from 3.3V
  Button      -> GPIO24 to GND  (internal pull-up enabled)

NOTE: All I2C devices must have pull-ups to 3.3V only. Remove onboard pull-up
resistors from the ADS1115 and MPU-6050 breakout boards if they are present,
or ensure all VCC pins are 3.3V (ADS1115 must stay on 5V for analog input range
— remove its onboard I2C pull-ups instead).

Install (Pi 5, Bookworm):
  sudo apt install -y python3-pil python3-smbus python3-gpiozero i2c-tools python3-lgpio
  pip3 install adafruit-blinka \
      adafruit-circuitpython-ssd1306 \
      adafruit-circuitpython-ads1x15 \
      adafruit-circuitpython-mpu6050 \
      --break-system-packages
  sudo raspi-config nonint do_i2c 0
  sudo usermod -aG gpio $USER
  # reboot, then verify: i2cdetect -y 1  (expect 3c, 48, 68)
"""

import math
import time
import threading
import board
import busio
from PIL import Image, ImageDraw, ImageFont
from gpiozero import Button
import adafruit_ssd1306
import adafruit_mpu6050
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
MPU6050_ADDR    = 0x68
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
mpu  = adafruit_mpu6050.MPU6050(i2c, address=MPU6050_ADDR)

font = ImageFont.load_default()

# ---------------------------------------------------------------------------
# IMU
# ---------------------------------------------------------------------------

def get_orientation(yaw_deg: float, dt: float) -> tuple[float, float, float]:
    """
    Returns (pitch, roll, yaw) in degrees.
    Pitch and roll from accelerometer (stable).
    Yaw from gyro integration (drifts — MPU-6050 has no magnetometer).
    """
    ax, ay, az = mpu.acceleration   # m/s²
    gx, gy, gz = mpu.gyro           # rad/s
    pitch = math.degrees(math.atan2(-ax, math.sqrt(ay**2 + az**2)))
    roll  = math.degrees(math.atan2(ay, az))
    yaw_deg += math.degrees(gz) * dt
    return pitch, roll, yaw_deg

# ---------------------------------------------------------------------------
# OLED
# ---------------------------------------------------------------------------

def draw_oled(v_batt: float, v_adc: float, btn_label: str,
              pitch: float, roll: float, yaw: float) -> None:
    img  = Image.new("1", (OLED_WIDTH, OLED_HEIGHT))
    draw = ImageDraw.Draw(img)

    draw.text((0,  0), f"BATT:{v_batt:6.2f}V  {btn_label}", font=font, fill=255)
    draw.text((0,  9), f"ADC: {v_adc*1000:6.1f} mV",        font=font, fill=255)
    draw.line([(0, 19), (OLED_WIDTH, 19)], fill=255, width=1)
    draw.text((0, 22), "--- ORIENTATION ---",                 font=font, fill=255)
    draw.text((0, 32), f"P: {pitch:+7.2f} deg",              font=font, fill=255)
    draw.text((0, 42), f"R: {roll:+7.2f} deg",               font=font, fill=255)
    draw.text((0, 52), f"Y: {yaw:+7.2f} deg",                font=font, fill=255)

    try:
        oled.image(img)
        oled.show()
    except OSError:
        print("[OLED] I2C write failed — transient bus error, skipping frame")


def clear_oled() -> None:
    try:
        oled.fill(0)
        oled.show()
    except OSError:
        pass

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    print("Breadboard test running. Ctrl-C to quit.")
    print(f"  OLED    : SSD1306 128x64 @ I2C 0x{OLED_ADDR:02X}")
    print(f"  ADC     : ADS1115 A1 @ I2C 0x{ADS1115_ADDR:02X}, gain=2/3 (±6.144 V), divider={ADC_DIVIDER}")
    print(f"  IMU     : MPU-6050 @ I2C 0x{MPU6050_ADDR:02X}  (yaw = gyro integration, drifts)")
    print(f"  Button  : GPIO{BUTTON_PIN} pull-up (active LOW)\n")

    yaw_deg       = 0.0
    last_time     = time.monotonic()
    btn_pressed_at = 0.0

    try:
        while True:
            now = time.monotonic()
            dt  = now - last_time
            last_time = now

            v_adc  = chan.voltage
            v_batt = v_adc / ADC_DIVIDER

            try:
                pitch, roll, yaw_deg = get_orientation(yaw_deg, dt)
            except OSError:
                print("[IMU] I2C read failed — transient bus error, skipping")
                pitch, roll = 0.0, 0.0

            if _btn_event.is_set():
                _btn_event.clear()
                btn_pressed_at = time.time()
                print(f"[BTN] GPIO{BUTTON_PIN} pressed  |  V_batt={v_batt:.2f}V  "
                      f"P={pitch:+.2f}  R={roll:+.2f}  Y={yaw_deg:+.2f}")

            btn_label = "PRESSED" if (time.time() - btn_pressed_at) < BTN_DISPLAY_S else "--"

            print(f"[ADC] V_adc={v_adc:.4f}V  V_batt={v_batt:.2f}V  |  "
                  f"[IMU] P={pitch:+.2f}  R={roll:+.2f}  Y={yaw_deg:+.2f}  BTN={btn_label}")

            draw_oled(v_batt, v_adc, btn_label, pitch, roll, yaw_deg)

            time.sleep(POLL_INTERVAL_S)

    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        clear_oled()
        button.close()


if __name__ == "__main__":
    main()
