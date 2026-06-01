#!/usr/bin/env python3
"""
dry_manual_software_testing.py

Hardware-in-the-loop dry test for the RAIV autonomous underwater scooter.

Peripherals tested:
  - ADS1115 ADC   (I2C 0x48): A0 -> battery 0 voltage, A1 -> battery 1 voltage
  - SSD1306 OLED  (I2C 0x3C, 128x64): displays batteries, mode, IMU orientation
  - MPU-6050 IMU  (I2C 0x68): pitch, roll (accel), yaw (gyro integration -- drifts)
  - GPIO 4  (pin 7 ): manual throttle button -- pull-up, active LOW
                      press toggles 60% throttle on main prop on/off
  - GPIO 24 (pin 18): autonomy mode button   -- pull-up, active LOW
                      press starts 20-second autonomy session
  - GPIO 12 (pin 32): main ESC PWM  (50 Hz, 1-2 ms)
  - GPIO 13 (pin 33): yaw  ESC PWM  (50 Hz, 1-2 ms)
  - Camera (J3)     : launched via Rpi_StrobeDetector.py subprocess, autonomy only

Wiring:
  ADS1115 A0  -> voltage divider output for battery 0
  ADS1115 A1  -> voltage divider output for battery 1
  OLED SDA    -> Pi GPIO2  (I2C-1 SDA)
  OLED SCL    -> Pi GPIO3  (I2C-1 SCL)
  MPU-6050    -> same I2C bus, addr 0x68, powered from 3.3V
  Buttons     -> GPIO pin to GND (internal pull-ups enabled)
  ESC signal  -> GPIO 12 / 13 (signal wire; ESCs powered separately)

ESC pulse mapping (gpiozero Servo, min_pulse_width=1ms, max_pulse_width=2ms):
  value = -1   ->  1.0 ms  ->  stopped / armed (zero throttle)
  value =  0   ->  1.5 ms  ->  neutral
  value =  0.2 ->  1.6 ms  ->  60% throttle  (MANUAL_THROTTLE_VALUE)
  value =  1   ->  2.0 ms  ->  full throttle

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
import os
import sys
import subprocess
import time
import threading
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

THROTTLE_BTN_PIN      = 4
AUTONOMY_BTN_PIN      = 24
MAIN_ESC_PIN          = 12
YAW_ESC_PIN           = 13

AUTONOMY_DURATION_S   = 20.0

# Servo.value maps -1..1 linearly to 1 ms..2 ms (50 Hz PWM).
# 60% throttle: value = -1 + 0.60*2 = 0.20  ->  1.60 ms pulse
MANUAL_THROTTLE_VALUE = 0.20
ESC_STOPPED_VALUE     = -1.0     # 1.0 ms -- armed / zero throttle

OLED_ADDR       = 0x3C
OLED_WIDTH      = 128
OLED_HEIGHT     = 64
ADS1115_ADDR    = 0x48
MPU6050_ADDR    = 0x68
ADC_GAIN        = 2 / 3  # ADS1115 PGA +/-6.144 V -- required for up to 4.8 V input

# Voltage divider ratio: V_adc = V_battery * ADC_DIVIDER  =>  V_battery = V_adc / ADC_DIVIDER
# Same resistor values on both battery lines, so one constant covers both.
# 24V * 0.20 = 4.8V  |  14V * 0.20 = 2.8V  -- both within ADS1115 +/-6.144V (gain=2/3)
ADC_DIVIDER     = 0.20

POLL_INTERVAL_S = 0.5
BTN_DISPLAY_S   = 1.0    # how long "AUTONOMY" flash stays on OLED after press

# IMU sanity check: accel magnitude should be close to 1g when stationary
G_EXPECTED      = 9.81   # m/s^2
G_TOLERANCE     = 1.5    # m/s^2 -- flag if outside this band

# Rpi_StrobeDetector.py lives one level up from this HITL/ folder
DETECTOR_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'Rpi_StrobeDetector.py')

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

_throttle_event = threading.Event()   # set by GPIO 4 press
_autonomy_event = threading.Event()   # set by GPIO 24 press

_throttle_on       = False
_autonomy_active   = False
_autonomy_end_time = 0.0
_detector_proc     = None   # Popen handle for Rpi_StrobeDetector.py

# ---------------------------------------------------------------------------
# Hardware init
# ---------------------------------------------------------------------------

throttle_btn = Button(THROTTLE_BTN_PIN, pull_up=True, bounce_time=0.05)
autonomy_btn = Button(AUTONOMY_BTN_PIN, pull_up=True, bounce_time=0.05)

throttle_btn.when_pressed = lambda: _throttle_event.set()
autonomy_btn.when_pressed = lambda: _autonomy_event.set()

i2c  = busio.I2C(board.SCL, board.SDA)
oled = adafruit_ssd1306.SSD1306_I2C(OLED_WIDTH, OLED_HEIGHT, i2c, addr=OLED_ADDR)
ads  = ADS.ADS1115(i2c, address=ADS1115_ADDR)
ads.gain = ADC_GAIN
chan_a0 = AnalogIn(ads, 0)   # A0: battery 0
chan_a1 = AnalogIn(ads, 1)   # A1: battery 1
mpu  = adafruit_mpu6050.MPU6050(i2c, address=MPU6050_ADDR)

# ESCs -- initial_value=-1 immediately sends 1 ms arming pulse on startup
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

def draw_oled(v_batt0: float, v_batt1: float, mode_label: str,
              pitch: float, roll: float, yaw: float) -> None:
    img  = Image.new("1", (OLED_WIDTH, OLED_HEIGHT))
    draw = ImageDraw.Draw(img)

    draw.text((0,  0), f"B0:{v_batt0:5.2f}V B1:{v_batt1:5.2f}V", font=font, fill=255)
    draw.text((0,  9), f"MODE: {mode_label}",                      font=font, fill=255)
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
# Autonomy mode -- strobe detector subprocess
# ---------------------------------------------------------------------------

def autonomy_start() -> None:
    global _detector_proc
    if not os.path.exists(DETECTOR_SCRIPT):
        print(f"[AUTO] WARNING: Rpi_StrobeDetector.py not found at {DETECTOR_SCRIPT}")
        return
    # --no-led: this script owns GPIO; detector must not touch it
    # --duration: match our autonomy window so it stops naturally
    _detector_proc = subprocess.Popen(
        [sys.executable, DETECTOR_SCRIPT,
         '--no-led', '--duration', str(int(AUTONOMY_DURATION_S))],
    )
    print(f"[AUTO] Rpi_StrobeDetector.py started (PID {_detector_proc.pid})")


def autonomy_stop() -> None:
    global _detector_proc
    if _detector_proc is None:
        return
    if _detector_proc.poll() is None:   # still running
        _detector_proc.terminate()
        try:
            _detector_proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            _detector_proc.kill()
    _detector_proc = None
    print("[AUTO] Strobe detector stopped")

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    global _throttle_on, _autonomy_active, _autonomy_end_time

    print("RAIV HITL dry test running. Ctrl-C to quit.")
    print(f"  OLED    : SSD1306 128x64 @ I2C 0x{OLED_ADDR:02X}")
    print(f"  ADC     : ADS1115 @ I2C 0x{ADS1115_ADDR:02X}, gain=2/3 (+/-6.144V), divider={ADC_DIVIDER}")
    print(f"            A0=batt0 (~24V nom)  |  A1=batt1 (~14V nom)")
    print(f"  IMU     : MPU-6050 @ I2C 0x{MPU6050_ADDR:02X}  (yaw = gyro integration, drifts)")
    print(f"  ESC     : main=GPIO{MAIN_ESC_PIN} (pin 32), yaw=GPIO{YAW_ESC_PIN} (pin 33)")
    print(f"  Buttons : GPIO{THROTTLE_BTN_PIN}=throttle toggle, "
          f"GPIO{AUTONOMY_BTN_PIN}=autonomy ({AUTONOMY_DURATION_S:.0f}s)")
    print(f"  Detector: {DETECTOR_SCRIPT}\n")

    startup_checks()

    print("[ESC] Sending 1 ms arming pulse -- waiting 2 s for ESCs to arm...")
    time.sleep(2.0)
    print("[ESC] Armed.\n")

    yaw_deg         = 0.0
    last_time       = time.monotonic()
    auto_pressed_at = 0.0   # for OLED "AUTONOMY" flash on button down

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

            # --- GPIO 4: manual throttle toggle ---
            if _throttle_event.is_set():
                _throttle_event.clear()
                if _autonomy_active:
                    print(f"[BTN{THROTTLE_BTN_PIN}] Ignored -- autonomy mode active")
                else:
                    _throttle_on = not _throttle_on
                    main_esc.value = MANUAL_THROTTLE_VALUE if _throttle_on else ESC_STOPPED_VALUE
                    print(f"[BTN{THROTTLE_BTN_PIN}] Manual throttle "
                          f"{'ON  (60%, 1.60 ms)' if _throttle_on else 'OFF (stopped, 1.00 ms)'}  |  "
                          f"B0={v_batt0:.2f}V  B1={v_batt1:.2f}V  "
                          f"P={pitch:+.2f}  R={roll:+.2f}  Y={yaw_deg:+.2f}")

            # --- GPIO 24: autonomy mode ---
            if _autonomy_event.is_set():
                _autonomy_event.clear()
                if _autonomy_active:
                    print(f"[BTN{AUTONOMY_BTN_PIN}] Autonomy already active -- ignoring")
                else:
                    _throttle_on       = False
                    main_esc.value     = ESC_STOPPED_VALUE
                    _autonomy_active   = True
                    _autonomy_end_time = time.monotonic() + AUTONOMY_DURATION_S
                    auto_pressed_at    = time.time()
                    autonomy_start()
                    print(f"[BTN{AUTONOMY_BTN_PIN}] *** AUTONOMY MODE ACTIVE -- "
                          f"{AUTONOMY_DURATION_S:.0f} s ***")

            # --- autonomy expiry check ---
            if _autonomy_active:
                remaining = _autonomy_end_time - time.monotonic()
                if remaining <= 0.0:
                    _autonomy_active = False
                    main_esc.value   = ESC_STOPPED_VALUE
                    yaw_esc.value    = ESC_STOPPED_VALUE
                    autonomy_stop()
                    print("[AUTO] *** Autonomy session ended -- returning to STANDBY ***")
                    remaining = 0.0
                else:
                    print(f"[AUTO] {remaining:4.1f}s left  |  "
                          f"B0={v_batt0:.2f}V  B1={v_batt1:.2f}V  "
                          f"P={pitch:+.2f}  R={roll:+.2f}  Y={yaw_deg:+.2f}")
            else:
                print(f"[ADC] B0={v_batt0:.2f}V (adc={v_adc0:.4f}V)  "
                      f"B1={v_batt1:.2f}V (adc={v_adc1:.4f}V)  |  "
                      f"[IMU] P={pitch:+.2f}  R={roll:+.2f}  Y={yaw_deg:+.2f}  |  "
                      f"THR={'ON' if _throttle_on else 'OFF'}")

            # --- OLED mode label ---
            if _autonomy_active:
                remaining = max(0.0, _autonomy_end_time - time.monotonic())
                mode_label = f"AUTO {remaining:.0f}s"
            elif (time.time() - auto_pressed_at) < BTN_DISPLAY_S:
                mode_label = "AUTONOMY"   # brief flash on button down before first poll
            elif _throttle_on:
                mode_label = "MAN 60%"
            else:
                mode_label = "--"

            draw_oled(v_batt0, v_batt1, mode_label, pitch, roll, yaw_deg)

            time.sleep(POLL_INTERVAL_S)

    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        main_esc.value = ESC_STOPPED_VALUE
        yaw_esc.value  = ESC_STOPPED_VALUE
        autonomy_stop()
        clear_oled()
        throttle_btn.close()
        autonomy_btn.close()
        main_esc.close()
        yaw_esc.close()


if __name__ == "__main__":
    main()
