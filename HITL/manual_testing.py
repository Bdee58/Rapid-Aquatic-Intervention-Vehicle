#!/usr/bin/env python3
"""
manual_testing.py

HITL manual-throttle test for the RAIV autonomous underwater scooter.
No autonomy mode, no camera. Both buttons drive the main prop by state
(throttle follows the button while it is held, not a toggle).

Button logic (state-based, polled every loop cycle):
  BTN1 (GPIO 4)  held alone  ->  40% forward  (1.70 ms)
  BTN2 (GPIO 24) held alone  ->  40% forward  (1.70 ms)
  Both held simultaneously   ->  80% forward  (1.90 ms)
  Neither held               ->   0% / neutral (1.50 ms)

Yaw ESC (GPIO 13) is locked at neutral (1.50 ms) throughout.

Peripherals tested:
  - ADS1115 ADC   (I2C 0x48): A0 -> battery 0 voltage, A1 -> battery 1 voltage
  - SSD1306 OLED  (I2C 0x3C, 128x64): batteries, throttle state, IMU orientation
  - MPU-6050 IMU  (I2C 0x68): pitch, roll (accel), yaw (gyro integration -- drifts)
  - GPIO 4  (pin 7 ): throttle button 1 -- pull-up, active LOW, read while held
  - GPIO 24 (pin 18): throttle button 2 -- pull-up, active LOW, read while held
  - GPIO 12 (pin 32): main ESC PWM  (50 Hz, 1-2 ms)
  - GPIO 13 (pin 33): yaw  ESC PWM  (50 Hz, 1-2 ms, locked at neutral)

Wiring:
  ADS1115 A0  -> voltage divider output for battery 0
  ADS1115 A1  -> voltage divider output for battery 1
  OLED SDA    -> Pi GPIO2  (I2C-1 SDA)
  OLED SCL    -> Pi GPIO3  (I2C-1 SCL)
  MPU-6050    -> same I2C bus, addr 0x68, powered from 3.3V
  Buttons     -> GPIO pin to GND (internal pull-ups enabled)
  ESC signal  -> GPIO 12 / 13 (signal wire; ESCs powered separately)

ESC pulse mapping (gpiozero Servo, min_pulse_width=1ms, max_pulse_width=2ms):
  Bidirectional ESC -- neutral = 1500us, arm at neutral.
  value =  0.0  ->  1.50 ms  ->  neutral / stopped / armed  (ESC_STOPPED_VALUE)
  value =  0.4  ->  1.70 ms  ->  40% forward  (THROTTLE_ONE_VALUE)
  value =  0.8  ->  1.90 ms  ->  80% forward  (THROTTLE_BOTH_VALUE)
  value =  1.0  ->  2.00 ms  ->  full forward
  40%: neutral(1500) + 0.40*(2000-1500) = 1700 us  =>  value = (1700-1500)/500 = 0.40
  80%: neutral(1500) + 0.80*(2000-1500) = 1900 us  =>  value = (1900-1500)/500 = 0.80

NOTE: All I2C devices must have pull-ups to 3.3V only. Remove onboard pull-up
resistors from the ADS1115 and MPU-6050 breakout boards if present,
or ensure all VCC pins are 3.3V (ADS1115 must stay on 5V for analog input range
-- remove its onboard I2C pull-ups instead).

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
import board
import busio
from PIL import Image, ImageDraw, ImageFont
from gpiozero import Button, Servo
import adafruit_ssd1306
import adafruit_mpu6050
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
ESC_STOPPED_VALUE   = 0.0   # 1.50 ms -- neutral, ESC arms here
THROTTLE_ONE_VALUE  = 0.4   # 1.70 ms -- 40% forward (one button held)
THROTTLE_BOTH_VALUE = 0.8   # 1.90 ms -- 80% forward (both buttons held)

OLED_ADDR       = 0x3C
OLED_WIDTH      = 128
OLED_HEIGHT     = 64
ADS1115_ADDR    = 0x48
MPU6050_ADDR    = 0x68
ADC_GAIN        = 2 / 3  # ADS1115 PGA +/-6.144 V -- required for up to 4.8 V input

# Voltage divider ratio: V_adc = V_battery * ADC_DIVIDER  =>  V_battery = V_adc / ADC_DIVIDER
# Same resistor values on both battery lines, so one constant covers both.
# To recalibrate: ADC_DIVIDER = (terminal adc= reading) / (DMM reading)
# Measured: adc=3.98V (19.9V displayed at 0.20), DMM=29.0V  =>  3.98/29.0 = 0.137
ADC_DIVIDER     = 0.137

POLL_INTERVAL_S = 0.05   # 20 Hz -- fast enough to feel responsive to button holds

# IMU sanity check: accel magnitude should be close to 1g when stationary
G_EXPECTED      = 9.81   # m/s^2
G_TOLERANCE     = 1.5    # m/s^2 -- flag if outside this band

# ---------------------------------------------------------------------------
# Hardware init
# ---------------------------------------------------------------------------

btn1 = Button(BTN1_PIN,  pull_up=True, bounce_time=0.05)
btn2 = Button(BTN2_PIN,  pull_up=True, bounce_time=0.05)

i2c  = busio.I2C(board.SCL, board.SDA)
oled = adafruit_ssd1306.SSD1306_I2C(OLED_WIDTH, OLED_HEIGHT, i2c, addr=OLED_ADDR)
ads  = ADS.ADS1115(i2c, address=ADS1115_ADDR)
ads.gain = ADC_GAIN
chan_a0 = AnalogIn(ads, 0)   # A0: battery 0
chan_a1 = AnalogIn(ads, 1)   # A1: battery 1
mpu  = adafruit_mpu6050.MPU6050(i2c, address=MPU6050_ADDR)

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
    """
    Ping every I2C peripheral and sanity-check the IMU before entering the
    main loop. Prints PASS / WARN / FAIL for each device.
    Returns True if everything looks good, False if any check failed.
    Script continues either way -- failures are warnings, not hard stops.
    """
    print("[INIT] Running startup checks...")
    all_ok = True

    # OLED
    try:
        oled.fill(0)
        oled.show()
        print(f"  [PASS] OLED    SSD1306  @ 0x{OLED_ADDR:02X}")
    except OSError as e:
        print(f"  [FAIL] OLED    SSD1306  @ 0x{OLED_ADDR:02X} -- {e}")
        all_ok = False

    # ADS1115 -- read both channels
    try:
        v_a0 = chan_a0.voltage
        v_a1 = chan_a1.voltage
        print(f"  [PASS] ADC     ADS1115  @ 0x{ADS1115_ADDR:02X}"
              f"  A0={v_a0:.3f}V  A1={v_a1:.3f}V")
    except OSError as e:
        print(f"  [FAIL] ADC     ADS1115  @ 0x{ADS1115_ADDR:02X} -- {e}")
        all_ok = False

    # MPU-6050 -- read accel and check magnitude
    try:
        ax, ay, az = mpu.acceleration
        g_mag = math.sqrt(ax**2 + ay**2 + az**2)
        if abs(g_mag - G_EXPECTED) <= G_TOLERANCE:
            print(f"  [PASS] IMU     MPU-6050 @ 0x{MPU6050_ADDR:02X}"
                  f"  |g|={g_mag:.2f} m/s^2  (expected ~{G_EXPECTED:.2f})")
        else:
            print(f"  [WARN] IMU     MPU-6050 @ 0x{MPU6050_ADDR:02X}"
                  f"  |g|={g_mag:.2f} m/s^2 -- expected ~{G_EXPECTED:.2f},"
                  f" check mounting / connection")
            all_ok = False
    except OSError as e:
        print(f"  [FAIL] IMU     MPU-6050 @ 0x{MPU6050_ADDR:02X} -- {e}")
        all_ok = False

    if all_ok:
        print("[INIT] All checks passed.\n")
    else:
        print("[INIT] WARNING: one or more checks failed -- continuing anyway.\n")

    return all_ok

# ---------------------------------------------------------------------------
# IMU
# ---------------------------------------------------------------------------

def get_orientation(yaw_deg: float, dt: float) -> tuple[float, float, float]:
    """
    Returns (pitch, roll, yaw) in degrees.
    Pitch and roll from accelerometer (stable).
    Yaw from gyro integration (drifts -- MPU-6050 has no magnetometer).
    """
    ax, ay, az = mpu.acceleration   # m/s^2
    gx, gy, gz = mpu.gyro           # rad/s
    pitch = math.degrees(math.atan2(-ax, math.sqrt(ay**2 + az**2)))
    roll  = math.degrees(math.atan2(ay, az))
    yaw_deg += math.degrees(gz) * dt
    return pitch, roll, yaw_deg

# ---------------------------------------------------------------------------
# OLED
# ---------------------------------------------------------------------------

def draw_oled(v_batt0: float, v_batt1: float, thr_label: str,
              pitch: float, roll: float, yaw: float) -> None:
    img  = Image.new("1", (OLED_WIDTH, OLED_HEIGHT))
    draw = ImageDraw.Draw(img)

    draw.text((0,  0), f"B0:{v_batt0:5.2f}V B1:{v_batt1:5.2f}V", font=font, fill=255)
    draw.text((0,  9), f"THR: {thr_label}",                        font=font, fill=255)
    draw.line([(0, 19), (OLED_WIDTH, 19)], fill=255, width=1)
    draw.text((0, 22), "--- ORIENTATION ---",                       font=font, fill=255)
    draw.text((0, 32), f"P: {pitch:+7.2f} deg",                    font=font, fill=255)
    draw.text((0, 42), f"R: {roll:+7.2f} deg",                     font=font, fill=255)
    draw.text((0, 52), f"Y: {yaw:+7.2f} deg",                      font=font, fill=255)

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
    print(f"  IMU     : MPU-6050 @ I2C 0x{MPU6050_ADDR:02X}  (yaw = gyro integration, drifts)")
    print(f"  ESC     : main=GPIO{MAIN_ESC_PIN} (pin 32), yaw=GPIO{YAW_ESC_PIN} (pin 33, locked)")
    print(f"  Buttons : GPIO{BTN1_PIN} + GPIO{BTN2_PIN}  (hold for throttle, both = 80%)\n")

    startup_checks()

    print("[ESC] Sending 1.5 ms neutral pulse -- waiting 3 s for bidirectional ESCs to arm...")
    time.sleep(3.0)
    print("[ESC] Armed. Yaw locked at neutral.\n")

    yaw_deg   = 0.0
    last_time = time.monotonic()
    last_thr  = ESC_STOPPED_VALUE   # track last commanded value to avoid redundant writes

    try:
        while True:
            now = time.monotonic()
            dt  = now - last_time
            last_time = now

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

            try:
                pitch, roll, yaw_deg = get_orientation(yaw_deg, dt)
            except OSError:
                print("[IMU] I2C read failed -- transient bus error, skipping")
                pitch, roll = 0.0, 0.0

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
                      f"B0={v_batt0:.2f}V  B1={v_batt1:.2f}V  "
                      f"P={pitch:+.2f}  R={roll:+.2f}  Y={yaw_deg:+.2f}")

            print(f"[ADC] B0={v_batt0:.2f}V (adc={v_adc0:.4f}V)  "
                  f"B1={v_batt1:.2f}V (adc={v_adc1:.4f}V)  |  "
                  f"[IMU] P={pitch:+.2f}  R={roll:+.2f}  Y={yaw_deg:+.2f}  |  "
                  f"THR={thr_label}  BTN=[{int(b1)}{int(b2)}]")

            draw_oled(v_batt0, v_batt1, thr_label, pitch, roll, yaw_deg)

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
