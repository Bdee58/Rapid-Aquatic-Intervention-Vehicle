#!/usr/bin/env python3
"""
manual_testing.py

HITL manual-throttle test for the RAIV autonomous underwater scooter.
No autonomy mode, no camera, no IMU. Both buttons drive the main prop by state
(throttle follows the button while it is held, not a toggle).

Button logic (state-based, polled every loop cycle):
  BTN1 (GPIO 4)  held alone  ->  40% forward  (1400 us)
  BTN2 (GPIO 24) held alone  ->  40% forward  (1400 us)
  Both held simultaneously   ->  80% forward  (1800 us)
  Neither held               ->  stopped       (1000 us)

Yaw ESC (GPIO 13) is locked at stopped (1000 us) throughout.

Peripherals tested:
  - ADS1115 ADC   (I2C 0x48): A0 -> battery 0 voltage, A1 -> battery 1 voltage
  - SSD1306 OLED  (I2C 0x3C, 128x64): batteries, throttle state
  - GPIO 4  (pin 7 ): throttle button 1 -- pull-up, active LOW, read while held
  - GPIO 24 (pin 18): throttle button 2 -- pull-up, active LOW, read while held
  - GPIO 12 (pin 32): main ESC PWM  (50 Hz, 1000-2000 us)
  - GPIO 13 (pin 33): yaw  ESC PWM  (50 Hz, locked at 1000 us)

Wiring:
  ADS1115 A0  -> voltage divider output for battery 0
  ADS1115 A1  -> voltage divider output for battery 1
  OLED SDA    -> Pi GPIO2  (I2C-1 SDA)
  OLED SCL    -> Pi GPIO3  (I2C-1 SCL)
  Buttons     -> GPIO pin to GND (internal pull-ups enabled)
  ESC signal  -> GPIO 12 / 13 signal wire + GND wire to Pi GND (required)

ESC pulse mapping (RPi.GPIO PWM, 50 Hz = 20 ms period):
  duty = pulse_us / 20000 * 100
  1000 us  ->  5.0 %  ->  stopped / arm signal  (ESC_STOPPED_US)
  1400 us  ->  7.0 %  ->  40% forward            (THROTTLE_ONE_US)
  1800 us  ->  9.0 %  ->  80% forward            (THROTTLE_BOTH_US)
  2000 us  -> 10.0 %  ->  full forward

Install (Pi 5, Bookworm):
  sudo apt install -y python3-pil python3-smbus python3-lgpio \
                      i2c-tools
  pip3 install adafruit-blinka \
      adafruit-circuitpython-ssd1306 \
      adafruit-circuitpython-ads1x15 \
      --break-system-packages
  sudo raspi-config nonint do_i2c 0
  sudo usermod -aG gpio $USER
  # reboot, then verify: i2cdetect -y 1  (expect 3c, 48)
"""

import time
import lgpio
import board
import busio
from PIL import Image, ImageDraw, ImageFont
import adafruit_ssd1306
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BTN1_PIN     = 4
BTN2_PIN     = 24
MAIN_ESC_PIN = 12
YAW_ESC_PIN  = 13

# ESC pulse widths in microseconds. duty = us / 20000 * 100 at 50 Hz.
ESC_STOPPED_US   = 1000   # 5.0% -- stopped, Apisqueen 100A arms here
THROTTLE_ONE_US  = 1400   # 7.0% -- 40% forward
THROTTLE_BOTH_US = 1800   # 9.0% -- 80% forward

OLED_ADDR       = 0x3C
OLED_WIDTH      = 128
OLED_HEIGHT     = 64
ADS1115_ADDR    = 0x48
ADC_GAIN        = 2 / 3  # ADS1115 PGA +/-6.144 V

# Voltage divider ratio: V_battery = V_adc / ADC_DIVIDER
# Measured: adc=3.98V at DMM=29.0V  =>  3.98/29.0 = 0.137
ADC_DIVIDER     = 0.137

POLL_INTERVAL_S = 0.05   # 20 Hz
GPIO_CHIP       = 4      # Pi 5 uses gpiochip4

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _duty(us: int) -> float:
    """Convert pulse width in microseconds to duty cycle % at 50 Hz."""
    return us / 20000.0 * 100.0

# ---------------------------------------------------------------------------
# Hardware init
# ---------------------------------------------------------------------------

h = lgpio.gpiochip_open(GPIO_CHIP)

# Buttons -- pull-up, active LOW
lgpio.gpio_claim_input(h, BTN1_PIN, lgpio.SET_PULL_UP)
lgpio.gpio_claim_input(h, BTN2_PIN, lgpio.SET_PULL_UP)

# ESC PWM -- 50 Hz, start at stopped signal
lgpio.gpio_claim_output(h, MAIN_ESC_PIN)
lgpio.gpio_claim_output(h, YAW_ESC_PIN)
lgpio.tx_pwm(h, MAIN_ESC_PIN, 50, _duty(ESC_STOPPED_US))
lgpio.tx_pwm(h, YAW_ESC_PIN,  50, _duty(ESC_STOPPED_US))

# I2C peripherals
i2c  = busio.I2C(board.SCL, board.SDA)
time.sleep(0.5)   # let I2C bus settle
oled = adafruit_ssd1306.SSD1306_I2C(OLED_WIDTH, OLED_HEIGHT, i2c, addr=OLED_ADDR)
ads  = ADS.ADS1115(i2c, address=ADS1115_ADDR)
ads.gain = ADC_GAIN
chan_a0 = AnalogIn(ads, 0)   # A0: battery 0
chan_a1 = AnalogIn(ads, 1)   # A1: battery 1

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
    Teaches the ESC where min (1000 us) and max (2000 us) are.
    Only needs to be done once -- ESC saves the range internally.
    """
    print("\n[CAL] ========== ESC CALIBRATION ==========")
    print("[CAL] ESC battery must be DISCONNECTED right now.")
    input("[CAL] Press Enter when ESC is unpowered and ready...")

    print("[CAL] Setting MAX throttle (2000 us, 10.0%)...")
    lgpio.tx_pwm(h, MAIN_ESC_PIN, 50, _duty(2000))
    lgpio.tx_pwm(h, YAW_ESC_PIN,  50, _duty(2000))

    print("[CAL] --> NOW connect the ESC battery.")
    print("[CAL]     Wait for beeps (cell-count beeps + long beep = entered cal mode).")
    input("[CAL] Press Enter once the ESC has beeped...")

    print("[CAL] Setting MIN throttle (1000 us, 5.0%)...")
    lgpio.tx_pwm(h, MAIN_ESC_PIN, 50, _duty(1000))
    lgpio.tx_pwm(h, YAW_ESC_PIN,  50, _duty(1000))

    print("[CAL]     Wait for confirmation beeps (1-2 short beeps).")
    input("[CAL] Press Enter once you hear the confirmation beeps...")

    print("[CAL] Calibration complete -- ESC now knows the throttle range.")
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
    print(f"  ADC     : ADS1115 @ I2C 0x{ADS1115_ADDR:02X}, gain=2/3, divider={ADC_DIVIDER}")
    print(f"            A0=batt0 (~24V nom)  |  A1=batt1 (~14V nom)")
    print(f"  ESC     : main=GPIO{MAIN_ESC_PIN} (pin 32), yaw=GPIO{YAW_ESC_PIN} (pin 33, locked)")
    print(f"  Buttons : GPIO{BTN1_PIN} + GPIO{BTN2_PIN}  (hold for throttle, both = 80%)\n")

    startup_checks()

    if input("Run ESC calibration? (y/N): ").strip().lower() == 'y':
        calibrate_escs()

    print(f"[ESC] Outputting {ESC_STOPPED_US} us ({_duty(ESC_STOPPED_US):.1f}%) -- power on ESC now.")
    print("[ESC] Listen for arm beep, then press Enter.")
    input("[ESC] Press Enter when ESC is armed: ")
    print("[ESC] Proceeding. Yaw locked.\n")

    last_thr_us = ESC_STOPPED_US

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
            b1 = lgpio.gpio_read(h, BTN1_PIN) == 0
            b2 = lgpio.gpio_read(h, BTN2_PIN) == 0

            if b1 and b2:
                thr_us    = THROTTLE_BOTH_US
                thr_label = "80%  [1+2]"
            elif b1 or b2:
                thr_us    = THROTTLE_ONE_US
                thr_label = f"40%  [{'1' if b1 else '2'}]"
            else:
                thr_us    = ESC_STOPPED_US
                thr_label = "0%"

            # only write to ESC when throttle level actually changes
            if thr_us != last_thr_us:
                lgpio.tx_pwm(h, MAIN_ESC_PIN, 50, _duty(thr_us))
                last_thr_us = thr_us
                print(f"[THR] {thr_label}  ({thr_us} us, {_duty(thr_us):.1f}%)  |  "
                      f"B0={v_batt0:.2f}V  B1={v_batt1:.2f}V")

            print(f"[ADC] B0={v_batt0:.2f}V (adc={v_adc0:.4f}V)  "
                  f"B1={v_batt1:.2f}V (adc={v_adc1:.4f}V)  |  "
                  f"THR={thr_label}  BTN=[{int(b1)}{int(b2)}]")

            draw_oled(v_batt0, v_batt1, thr_label)

            time.sleep(POLL_INTERVAL_S)

    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        lgpio.tx_pwm(h, MAIN_ESC_PIN, 50, _duty(ESC_STOPPED_US))
        lgpio.tx_pwm(h, YAW_ESC_PIN,  50, _duty(ESC_STOPPED_US))
        clear_oled()
        lgpio.gpiochip_close(h)


if __name__ == "__main__":
    main()
