"""
Bluetooth packet format for the Pulse Sensor System (BMET2922 Major Project).

One packet is sent by the ESP32 every second (Reqt 18). It is plain ASCII text,
modelled on NMEA 0183 GPS sentences:

    $SSSSS,HHH.H,R,s01,s02,...,s50*CC\\n

    $       start character - marks the start of a packet so the host can
            re-synchronise after partial or corrupted data
    SSSSS   sequence number, 5 digits, 00000-65535 then wraps (uint16 on MCU)
    HHH.H   MCU pulse rate in BPM, 1 decimal place, 000.0 = no valid pulse
    R       recording state from the hardware button, 0 or 1 (the current
            state, not a toggle event, so a lost packet cannot desync it)
    sNN     50 raw ADC samples, 4 digits each (0000-4095), oldest first,
            one every 20 ms = the last second of data
    *       end of data, checksum follows
    CC      checksum: XOR of every byte between '$' and '*' (exclusive),
            as 2 upper-case hex digits (the NMEA convention)
    \\n      end of packet

Every field is zero-padded to a fixed width, so every packet is exactly
PACKET_LEN = 268 bytes (Reqt 18: all packets the same length).

Why text rather than binary: it can be read directly in the Arduino Serial
Monitor while debugging, there is no little-/big-endian ambiguity between the
ESP32 and the host, and '\\n' plus '$' give simple framing. The cost is size
(268 B vs ~108 B binary), which is negligible for Bluetooth SPP at 1 packet/s.

The ESP32 side of this format is in Code/esp32/bt_packet_sender/packet.cpp.
Both sides must be changed together.

Author: Bianca Dibden
Date: 15-Oct-2026
"""

from __future__ import annotations

from dataclasses import dataclass

START = b"$"
CHECKSUM_SEP = b"*"
END = b"\n"
SAMPLES_PER_PACKET = 50
SEQ_MODULO = 65536
ADC_MAX = 4095
HR_MAX = 999.9

# 1 ($) + 6 (SSSSS,) + 6 (HHH.H,) + 2 (R,) + 50*4 + 49 commas + 1 (*) + 2 (CC) + 1 (\n)
PACKET_LEN = 1 + 6 + 6 + 2 + SAMPLES_PER_PACKET * 4 + (SAMPLES_PER_PACKET - 1) + 1 + 2 + 1
assert PACKET_LEN == 268


class PacketError(ValueError):
    """
    Raised when a received line is not a valid packet. str(error) is a short
    reason suitable for the system log, e.g. 'bad checksum'.

    Author: Bianca Dibden
    """


@dataclass
class Packet:
    """
    One decoded packet.

    Attributes
    ----------
    seq : int
        Sequence number (0-65535).
    mcu_hr : float
        Pulse rate calculated on the ESP32 (BPM); 0.0 means no valid pulse.
    recording : bool
        Recording state from the hardware button.
    samples : list of int
        50 raw ADC samples, oldest first.

    Author: Bianca Dibden
    """
    seq: int
    mcu_hr: float
    recording: bool
    samples: list


def checksum(payload: bytes) -> int:
    """
    XOR of all bytes in payload (the bytes between '$' and '*').

    Parameters
    ----------
    payload : bytes

    Returns
    -------
    int
        0-255.

    Author: Bianca Dibden
    """
    c = 0
    for b in payload:
        c ^= b
    return c


def build_packet(seq: int, mcu_hr: float, recording: bool, samples) -> bytes:
    """
    Build a packet exactly as the ESP32 does. Used by the simulator and tests.

    Parameters
    ----------
    seq : int
        Sequence number (taken modulo 65536).
    mcu_hr : float
        MCU pulse rate (BPM), clamped to 0-999.9.
    recording : bool
        Recording state.
    samples : sequence of int
        Exactly 50 samples, clamped to 0-4095.

    Returns
    -------
    bytes
        A PACKET_LEN-byte packet ending in '\\n'.

    Author: Bianca Dibden
    """
    if len(samples) != SAMPLES_PER_PACKET:
        raise ValueError(f"need {SAMPLES_PER_PACKET} samples, got {len(samples)}")
    hr = min(max(float(mcu_hr), 0.0), HR_MAX)
    fields = [f"{seq % SEQ_MODULO:05d}", f"{hr:05.1f}", "1" if recording else "0"]
    fields += [f"{min(max(int(s), 0), ADC_MAX):04d}" for s in samples]
    payload = ",".join(fields).encode("ascii")
    pkt = START + payload + CHECKSUM_SEP + f"{checksum(payload):02X}".encode("ascii") + END
    assert len(pkt) == PACKET_LEN
    return pkt


def parse_packet(line: bytes) -> Packet:
    """
    Validate and decode one packet.

    Checks, in order: length, start character, checksum separator, checksum,
    field count, field formats, and value ranges. The first failure raises
    PacketError with a short reason.

    Parameters
    ----------
    line : bytes
        One packet, with or without the trailing '\\n' (a trailing '\\r' is
        also tolerated).

    Returns
    -------
    Packet

    Raises
    ------
    PacketError

    Author: Bianca Dibden
    """
    line = line.rstrip(b"\r\n")
    if len(line) != PACKET_LEN - 1:
        raise PacketError(f"wrong length ({len(line)} bytes, expected {PACKET_LEN - 1})")
    if line[:1] != START:
        raise PacketError("missing start character")
    if line[-3:-2] != CHECKSUM_SEP:
        raise PacketError("missing checksum separator")
    payload, cs_text = line[1:-3], line[-2:]
    try:
        cs_rx = int(cs_text, 16)
    except ValueError:
        raise PacketError("checksum not hex") from None
    if cs_rx != checksum(payload):
        raise PacketError("bad checksum")

    fields = payload.split(b",")
    if len(fields) != 3 + SAMPLES_PER_PACKET:
        raise PacketError(f"wrong field count ({len(fields)})")
    try:
        seq = int(fields[0])
        mcu_hr = float(fields[1])
        rec = fields[2]
        samples = [int(f) for f in fields[3:]]
    except ValueError:
        raise PacketError("non-numeric field") from None
    if rec not in (b"0", b"1"):
        raise PacketError("bad recording flag")
    if not 0 <= seq < SEQ_MODULO:
        raise PacketError("sequence out of range")
    if not 0.0 <= mcu_hr <= HR_MAX:
        raise PacketError("MCU HR out of range")
    if any(s < 0 or s > ADC_MAX for s in samples):
        raise PacketError("sample out of ADC range")
    return Packet(seq=seq, mcu_hr=mcu_hr, recording=rec == b"1", samples=samples)
