# Pulse Sensor System — BMET2922 Major Project

Wearable PPG pulse monitor: ESP32 + PPG sensor → Bluetooth SPP → Python host.

## Folders
| Folder | Contents |
| --- | --- |
| `A2_Lumi/` | **Final LumiPulse sketch** (`A2_Lumi.ino` + `switch`/`packet` files): local monitor, button (3 s hold toggles buzzer sound), LEDs, LCD, Bluetooth packets (`AbiESP32`) |
| `archive/` | Superseded sketches kept for reference: `FinalLumiPulse/`, `PULSENEW/`, `pulse_monitor/` (replaced by `A2_Lumi/`), `bt_packet_sender/` (early Bluetooth test sketch), and the old `gui.py` |
| `host/` | Python host: `comms.py` (Bluetooth receiver), `ppg_dsp.py` (filtering + heart rate), `packet_format.py`, `sim_packet.py` (fake ESP32), tests |
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
50 raw ADC samples, XOR checksum. See `host/packet_format.py` and `A2_Lumi/packet.h`.

## Alarm limits (host → ESP32)
`$LIM,LLL.L,HHH.H*CC\n` — the low/high HR limits set in the GUI (Apply Limits). Sent on connect, on change and once a second; the ESP32 alarm, buzzer and LCD LOW/HIGH use them (defaults 50/100 BPM until received) and the serial monitor prints `LowLimit` and `HighLimit` with each sample.

Authors: Bianca Dibden (comms + signal processing), Abi, Alva
