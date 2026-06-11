#!/usr/bin/env python3
"""
manual_testing.py

HITL manual-throttle test for the RAIV autonomous underwater scooter.
No autonomy mode, no camera, no IMU. Both buttons drive the main prop by state
(throttle follows the button while it is held, not a toggle).

Button logic (state-based, polled every loop cycle):
  BTN1 (GPIO 4)  held alone  ->  40% forward  (1.70 ms)
  BTN2 (GPIO 24) held alone  ->  40% forward  (1.70 ms)
  Both held simultaneously   ->  80% forward  (1.90 ms)
  Neither held               ->   0% / neutral (1.50 ms)

Yaw ESC (GPIO 13) is locked at neutral (1.50 ms) throughout.

Peripherals tested:
  - ADS1115 ADC   (I2C 0x48): A0 -> battery 0 voltage, A1 -> battery 1 voltage
  - SSD1306 OLED  (I2C 0x3C, 128x64): batteries, throttle state
  - GPIO 4  (pin 7 ): throttle button 1 -- pull-up, active LOW, read while held
  - GPIO 24 (pin 18): throttle button 2 -- pull-up, active LOW, read while held
  - GPIO 12 (pin 32): main ESC PWM  (50 Hz, 1-2 ms)
  - GPIO 13 (pin 33): yaw  ESC PWM  (50 Hz, 1-2 ms, locked at neutral)

Wiring:
  ADS1115 A0  -> voltage divider output for battery 0
  ADS1115 A1  -> voltage divider output for battery 1
  OLED SDA    -> Pi GPIO2  (I2C-1 SDA)
  OLED SCL    -> Pi GPIO3  (I2C-1 SCL)
  Buttons     -> GPIO pin to GND (internal pull-ups enabled)
  ESC signal  -> GPIO 12 / 13 (signal wire; ESCs powered separately)

ESC pulse mapping (gpiozero Servo, min_pulse_width=1ms, max_pulse_width=2ms):
  Bidirectional ESC -- neutral = 1500us, arm at neutral.
  value = -1.0  ->  1.00 ms  ->  min throttle / armed        (ESC_STOPPED_VALUE)
  value = -0.2  ->  1.40 ms  ->  40% of 1000-2000 us range  (THROTTLE_ONE_VALUE)
  value =  0.6  ->  1.80 ms  ->  80% of 1000-2000 us range  (THROTTLE_BOTH_VALUE)
  value =  1.0  ->  2.00 ms  ->  full throttle
  40%: 1000 + 0.40*1000 = 1400 us  =>  value = (1400-1500)/500 = -0.20
  80%: 1000 + 0.80*1000 = 1800 us  =>  value = (1800-1500)/500 =  0.60

Install (Pi 5, Bookworm):
  sudo apt install -y python3-pil python3-smbus python3-gpiozero i2c-tools python3-lgpio
  pip3 install adafruit-blinka \
      adafruit-circuitpython-ssd1306 \
      adafruit-circuitpython-ads1x15 \
      --break-system-packages
  sudo raspi-config nonint do_i2c 0
  sudo usermod -aG gpio $USER
  # reboot, then verify: i2cdetect -y 1  (expect 3c, 48)
"""

import time
import board
import busio
from PIL import Image, ImageDraw, ImageFont
from gpiozero import Button, Servo
import adafruit_ssd1306
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BTN1_PIN        = 4
BTN2_PIN        = 24
MAIN_ESC_PIN    = 12
YAW_ESC_PIN     = 13

# Servo.value maps -1..1 linearly to 1 ms..2 ms (50 Hz PWM).
# Bidirectional ESC: neutral = 1500 us (value=0.0), forward range 1500-2000 us.
ESC_STOPPED_VALUE   = -1.0  # 1.00 ms -- min throttle, standard unidirectional ESC arms here
THROTTLE_ONE_VALUE  = -0.2  # 1.40 ms -- 40% of 1000-2000 us range
THROTTLE_BOTH_VALUE =  0.6  # 1.80 ms -- 80% of 1000-2000 us range

OLED_ADDR       = 0x3C
OLED_WIDTH      = 128
OLED_HEIGHT     = 64
ADS1115_ADDR    = 0x48
ADC_GAIN        = 2 / 3  # ADS1115 PGA +/-6.144 V -- required for up to 4.8 V input

# Voltage divider ratio: V_adc = V_battery * ADC_DIVIDER  =>  V_battery = V_adc / ADC_DIVIDER
# Same resistor values on both battery lines, so one constant covers both.
# To recalibrate: ADC_DIVIDER = (terminal adc= reading) / (DMM reading)
# Measured: adc=3.98V (19.9V displayed at 0.20), DMM=29.0V  =>  3.98/29.0 = 0.137
ADC_DIVIDER     = 0.137

POLL_INTERVAL_S = 0.05   # 20 Hz -- fast enough to feel responsive to button holds

# ---------------------------------------------------------------------------
# Hardware init
# ---------------------------------------------------------------------------

btn1 = Button(BTN1_PIN,  pull_up=True, bounce_time=0.05)
btn2 = Button(BTN2_PIN,  pull_up=True, bounce_time=0.05)

i2c  = busio.I2C(board.SCL, board.SDA)
time.sleep(0.5)   # let I2C bus settle before hitting devices
oled = adafruit_ssd1306.SSD1306_I2C(OLED_WIDTH, OLED_HEIGHT, i2c, addr=OLED_ADDR)
ads  = ADS.ADS1115(i2c, address=ADS1115_ADDR)
ads.gain = ADC_GAIN
chan_a0 = AnalogIn(ads, 0)   # A0: battery 0
chan_a1 = AnalogIn(ads, 1)   # A1: battery 1

# ESCs -- initial_value=ESC_STOPPED_VALUE sends 1.5 ms neutral on startup for arming
main_esc = Servo(MAIN_ESC_PIN, initial_value=ESC_STOPPED_VALUE,
                 min_pulse_width=1/1000, max_pulse_width=2/1000)
yaw_esc  = Servo(YAW_ESC_PIN,  initial_value=ESC_STOPPED_VALUE,
                 min_pulse_width=1/1000, max_pulse_width=2/1000)

font = ImageFont.load_default()

# ---------------------------------------------------------------------------
# Startup health checks
# ---------------------------------------------------------------------------

def startup_checks() -> bool:
    print("[INIT] Running startup checks...")
    all_ok = True

    try:
        oled.fill(0)
        oled.show()
        print(f"  [PASS] OLED    SSD1306  @ 0x{OLED_ADDR:02X}")
    except OSError as e:
        print(f"  [FAIL] OLED    SSD1306  @ 0x{OLED_ADDR:02X} -- {e}")
        all_ok = False

    try:
        v_a0 = chan_a0.voltage
        v_a1 = chan_a1.voltage
        print(f"  [PASS] ADC     ADS1115  @ 0x{ADS1115_ADDR:02X}"
              f"  A0={v_a0:.3f}V  A1={v_a1:.3f}V")
    except OSError as e:
        print(f"  [FAIL] ADC     ADS1115  @ 0x{ADS1115_ADDR:02X} -- {e}")
        all_ok = False

    if all_ok:
        print("[INIT] All checks passed.\n")
    else:
        print("[INIT] WARNING: one or more checks failed -- continuing anyway.\n")

    return all_ok

# ---------------------------------------------------------------------------
# ESC calibration
# ---------------------------------------------------------------------------

def calibrate_escs() -> None:
    """
    One-time throttle range calibration. ESC must be UNPOWERED at the start.
    Teaches the ESC where min (1000us) and max (2000us) are so it arms correctly.
    Only needs to be done once per ESC -- it saves the range internally.
    """
    print("\n[CAL] ========== ESC CALIBRATION ==========")
    print("[CAL] ESC battery must be DISCONNECTED right now.")
    input("[CAL] Press Enter when ESC is unpowered and ready...")

    print("[CAL] Setting MAX throttle (2.00 ms) on both ESCs...")
    main_esc.value = 1.0
    yaw_esc.value  = 1.0

    print("[CAL] --> NOW connect the ESC battery.")
    print("[CAL]     Wait for the beep sequence (usually cell-count beeps + long beep).")
    input("[CAL] Press Enter once the ESC has finished its startup beeps...")

    print("[CAL] Setting MIN throttle (1.00 ms)...")
    main_esc.value = -1.0
    yaw_esc.value  = -1.0

    print("[CAL]     Wait for the ESC confirmation beeps (1-2 short beeps).")
    input("[CAL] Press Enter once you hear the confirmation beeps...")

    print("[CAL] Calibration complete -- ESC now knows the full throttle range.")
    print("[CAL] ==========================================\n")

# ---------------------------------------------------------------------------
# OLED
# ---------------------------------------------------------------------------

def draw_oled(v_batt0: float, v_batt1: float, thr_label: str) -> None:
    img  = Image.new("1", (OLED_WIDTH, OLED_HEIGHT))
    draw = ImageDraw.Draw(img)

    draw.text((0,  0), f"B0:{v_batt0:5.2f}V B1:{v_batt1:5.2f}V", font=font, fill=255)
    draw.line([(0, 10), (OLED_WIDTH, 10)], fill=255, width=1)
    draw.text((0, 13), "THROTTLE:",                                 font=font, fill=255)
    draw.text((0, 26), f"  {thr_label}",                           font=font, fill=255)

    try:
        oled.image(img)
        oled.show()
    except OSError:
        print("[OLED] I2C write failed -- transient bus error, skipping frame")


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
    print("RAIV HITL manual test running. Ctrl-C to quit.")
    print(f"  OLED    : SSD1306 128x64 @ I2C 0x{OLED_ADDR:02X}")
    print(f"  ADC     : ADS1115 @ I2C 0x{ADS1115_ADDR:02X}, gain=2/3 (+/-6.144V), divider={ADC_DIVIDER}")
    print(f"            A0=batt0 (~24V nom)  |  A1=batt1 (~14V nom)")
    print(f"  ESC     : main=GPIO{MAIN_ESC_PIN} (pin 32), yaw=GPIO{YAW_ESC_PIN} (pin 33, locked)")
    print(f"  Buttons : GPIO{BTN1_PIN} + GPIO{BTN2_PIN}  (hold for throttle, both = 80%)\n")

    startup_checks()

    if input("Run ESC calibration? (y/N): ").strip().lower() == 'y':
        calibrate_escs()

    print("[ESC] Sending 1.00 ms min-throttle pulse -- waiting 3 s for ESCs to arm...")
    time.sleep(3.0)
    print("[ESC] Armed. Yaw locked at neutral.\n")

    last_thr = ESC_STOPPED_VALUE   # track last commanded value to avoid redundant writes

    try:
        while True:
            # --- sensor reads ---
            try:
                v_adc0  = chan_a0.voltage
                v_batt0 = v_adc0 / ADC_DIVIDER
            except OSError:
                print("[ADC] A0 I2C read failed -- transient bus error, skipping")
                v_adc0, v_batt0 = 0.0, 0.0

            try:
                v_adc1  = chan_a1.voltage
                v_batt1 = v_adc1 / ADC_DIVIDER
            except OSError:
                print("[ADC] A1 I2C read failed -- transient bus error, skipping")
                v_adc1, v_batt1 = 0.0, 0.0

            # --- throttle state: read both buttons simultaneously ---
            b1 = btn1.is_pressed
            b2 = btn2.is_pressed

            if b1 and b2:
                thr_value = THROTTLE_BOTH_VALUE
                thr_label = "80%  [1+2]"
            elif b1 or b2:
                thr_value = THROTTLE_ONE_VALUE
                thr_label = f"40%  [{'1' if b1 else '2'}]"
            else:
                thr_value = ESC_STOPPED_VALUE
                thr_label = "0%"

            # only write to ESC when throttle level actually changes
            if thr_value != last_thr:
                main_esc.value = thr_value
                last_thr = thr_value
                print(f"[THR] {thr_label}  ({thr_value:.2f} -> {1500 + thr_value*500:.0f} us)  |  "
                      f"B0={v_batt0:.2f}V  B1={v_batt1:.2f}V")

            print(f"[ADC] B0={v_batt0:.2f}V (adc={v_adc0:.4f}V)  "
                  f"B1={v_batt1:.2f}V (adc={v_adc1:.4f}V)  |  "
                  f"THR={thr_label}  BTN=[{int(b1)}{int(b2)}]")

            draw_oled(v_batt0, v_batt1, thr_label)

            time.sleep(POLL_INTERVAL_S)

    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        main_esc.value = ESC_STOPPED_VALUE
        yaw_esc.value  = ESC_STOPPED_VALUE
        clear_oled()
        btn1.close()
        btn2.close()
        main_esc.close()
        yaw_esc.close()


if __name__ == "__main__":
    main()
