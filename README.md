# Pulse Sensor System — BMET2922 Major Project

Wearable PPG pulse monitor: ESP32 + PPG sensor → Bluetooth SPP → Python host.

## Folders
| Folder | Contents |
| --- | --- |
| `esp32/pulse_monitor/` | **Main ESP32 sketch**: Abi's acquisition, MCU HR, button, LEDs, buzzer, LCD + Bluetooth packets (`LumiPulse-G01`). Set `USE_REAL_SENSOR 0` to replay pulse.csv without the sensor |
| `esp32/bt_packet_sender/` | Standalone Bluetooth test sketch: 50 Hz sampling, 1 packet/s over Bluetooth (open `bt_packet_sender.ino` in the Arduino IDE) |
| `host/` | Python host: `comms.py` (Bluetooth receiver), `ppg_dsp.py` (filtering + heart rate), `packet_format.py`, `packet_sim.py` (fake ESP32), tests |
| `data/` | `pulse.csv` — course PPG recording used for replay and tests |

## Setup (Windows)
```
py -m pip install -r requirements.txt
cd host
py test_comms.py --quick        # checks everything works, no hardware needed
py comms.py SIM                 # live view using the simulated ESP32
py comms.py --list              # find the ESP32's Bluetooth COM port
py comms.py COM5                # live view from the real ESP32 (use your COM number)
```
On Mac/Linux use `python3` instead of `py`.

## Packet format (ESP32 → host, 1 per second, always 268 bytes)
`$SSSSS,HHH.H,R,s01,...,s50*CC\n` — sequence number, MCU pulse rate (BPM), recording state,
50 raw ADC samples, XOR checksum. See `host/packet_format.py` and `esp32/bt_packet_sender/packet.h`.

Authors: Bianca Dibden (comms + signal processing), Abi, Alva
