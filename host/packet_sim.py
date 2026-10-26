"""
Simulated ESP32 for developing and testing the host without hardware.

SimulatedDevice behaves like the real ESP32 + Bluetooth SPP port:
  - it produces one packet per period (default 1 s) in the real packet format,
    replaying Code/data/pulse.csv as the sensor signal and its pulse-rate
    column as the "MCU HR";
  - open() returns a serial-like object (read / in_waiting / close / is_open),
    so PPGLink can use it in place of pyserial: PPGLink("SIM") or
    PPGLink(port, serial_factory=device.open);
  - faults can be injected: dropped, corrupted, reordered packets, junk bytes,
    power loss and MCU reset (sequence restarts at 0).

It can also drive a pseudo-terminal (Mac/Linux) so a separate program, e.g.
the GUI, can open it exactly like the real Bluetooth port:

    python3 packet_sim.py --pty                 # prints a port name to open
    python3 packet_sim.py --pty --drop 0.05 --corrupt 0.05

Author: Bianca Dibden
Date: 15-Oct-2026
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import threading
import time

import packet_format as pf

try:
    from serial import SerialException
except ImportError:
    class SerialException(OSError):
        """Stand-in when pyserial is not installed. Author: Bianca Dibden"""

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(HERE, "..", "data", "pulse.csv")


def load_replay_data(path=DEFAULT_CSV):
    """
    Load pulse.csv as replay data.

    Returns
    -------
    samples : list of int
        Raw ADC samples.
    rates : list of float
        Pulse rate for each sample, converted from Hz to BPM.

    Author: Bianca Dibden
    """
    samples, rates = [], []
    with open(path) as f:
        for row in csv.reader(f):
            samples.append(int(row[0]))
            rates.append(float(row[1]) * 60.0)
    return samples, rates


class SimulatedDevice:
    """
    A simulated ESP32 that generates packets in real time.

    Packet k (k = 0, 1, 2 ...) becomes available period*(k+1) seconds after
    power-on, has sequence number k mod 65536, and carries samples
    50k .. 50k+49 of the (looped) replay data.

    Fault-injection attributes (probabilities 0-1, applied per packet):
        drop_prob, corrupt_prob, swap_prob, junk_prob

    Author: Bianca Dibden
    """

    def __init__(self, period=1.0, samples=None, rates=None, seed=0,
                 drop_prob=0.0, corrupt_prob=0.0, swap_prob=0.0, junk_prob=0.0,
                 loss_mode="silent"):
        """
        Parameters
        ----------
        period : float
            Seconds per packet (1.0 = real time; smaller to speed up tests).
        samples, rates : list, optional
            Replay data; defaults to pulse.csv.
        seed : int
            Random seed for reproducible fault injection.
        drop_prob, corrupt_prob, swap_prob, junk_prob : float
            Per-packet fault probabilities.
        loss_mode : str
            How a broken link looks to the host: 'silent' (open port, no
            data - typical of Bluetooth SPP) or 'error' (read() raises).

        Author: Bianca Dibden
        """
        if samples is None:
            samples, rates = load_replay_data()
        self.samples, self.rates = samples, rates
        self.period = period
        self.rng = random.Random(seed)
        self.drop_prob, self.corrupt_prob = drop_prob, corrupt_prob
        self.swap_prob, self.junk_prob = swap_prob, junk_prob
        self.loss_mode = loss_mode
        self.recording = False
        self.lock = threading.Lock()
        self.powered = True
        self.t_on = time.monotonic()
        self.generation = 0      # increments on power-off/reset; kills open connections
        self.packets_built = 0
        self.received = []       # byte strings written by the host (alarm messages)
        self.held = None         # packet held back to simulate out-of-order delivery

    # --- controls used by tests --------------------------------------------
    def power_off(self):
        """
        Simulate the ESP32 losing power / going out of range.

        Author: Bianca Dibden
        """
        with self.lock:
            self.powered = False
            self.generation += 1

    def power_on(self):
        """
        Simulate the ESP32 booting: sequence restarts at 0.

        Author: Bianca Dibden
        """
        with self.lock:
            self.powered = True
            self.t_on = time.monotonic()

    def reset(self, downtime=0.5):
        """
        Simulate pressing the ESP32 reset button.

        Author: Bianca Dibden
        """
        self.power_off()
        time.sleep(downtime)
        self.power_on()

    # --- packet generation --------------------------------------------------
    def packets_due(self):
        """
        Number of packets produced since power-on.

        Author: Bianca Dibden
        """
        return int((time.monotonic() - self.t_on) / self.period)

    def build(self, k):
        """
        Build packet k (no faults applied).

        Author: Bianca Dibden
        """
        n = len(self.samples)
        idx = [(k * pf.SAMPLES_PER_PACKET + i) % n for i in range(pf.SAMPLES_PER_PACKET)]
        chunk = [self.samples[i] for i in idx]
        hr = self.rates[idx[-1]] if self.rates else 0.0
        self.packets_built += 1
        return pf.build_packet(k, hr, self.recording, chunk)

    def faulty_stream(self, packets):
        """
        Apply the configured faults to a list of packets.

        Returns
        -------
        bytes

        Author: Bianca Dibden
        """
        out = []
        for p in packets:
            r = self.rng.random
            if r() < self.drop_prob:
                continue
            if r() < self.corrupt_prob:
                b = bytearray(p)
                i = self.rng.randrange(1, len(b) - 4)
                b[i] = ord("7") if b[i] != ord("7") else ord("3")
                p = bytes(b)
            if r() < self.junk_prob:
                out.append(bytes(self.rng.randrange(32, 127) for _ in range(self.rng.randrange(3, 40))))
            if self.held is not None:
                # deliver the held-back packet after this one -> out of order
                out.append(p)
                out.append(self.held)
                self.held = None
                continue
            if r() < self.swap_prob:
                self.held = p
                continue
            out.append(p)
        return b"".join(out)

    def open(self, port="SIM", baudrate=115200, timeout=0.1):
        """
        Open a connection (pyserial-style factory used by PPGLink).

        Raises
        ------
        SerialException
            If the device is powered off (like a Bluetooth port with no device).

        Author: Bianca Dibden
        """
        with self.lock:
            if not self.powered:
                raise SerialException(f"could not open port {port}: device not available")
            return SimulatedSerial(self, timeout)


class SimulatedSerial:
    """
    Minimal pyserial.Serial look-alike connected to a SimulatedDevice.

    Author: Bianca Dibden
    """

    def __init__(self, device, timeout):
        """Author: Bianca Dibden"""
        self.dev = device
        self.timeout = timeout
        self.generation = device.generation
        self.next_k = device.packets_due()   # only packets produced after connecting
        self.buf = bytearray()
        self.is_open = True

    def _alive(self):
        """True if the link has not been broken since opening. Author: Bianca Dibden"""
        return self.dev.powered and self.dev.generation == self.generation

    def _pump(self):
        """
        Move newly produced packets into the receive buffer.

        Author: Bianca Dibden
        """
        if not self.is_open:
            raise SerialException("port is closed")
        if not self._alive():
            if self.dev.loss_mode == "error":
                raise SerialException("device disconnected")
            return
        due = self.dev.packets_due()
        if due > self.next_k:
            pkts = [self.dev.build(k) for k in range(self.next_k, due)]
            self.next_k = due
            self.buf.extend(self.dev.faulty_stream(pkts))

    @property
    def in_waiting(self):
        """Bytes waiting to be read. Author: Bianca Dibden"""
        self._pump()
        return len(self.buf)

    def read(self, n=1):
        """
        Read up to n bytes, waiting at most timeout seconds for data.

        Author: Bianca Dibden
        """
        deadline = time.monotonic() + (self.timeout or 0)
        while True:
            self._pump()
            if self.buf:
                data = bytes(self.buf[:n])
                del self.buf[:n]
                return data
            if time.monotonic() >= deadline:
                return b""
            time.sleep(0.005)

    def reset_input_buffer(self):
        """Discard buffered input. Author: Bianca Dibden"""
        self.buf.clear()

    def write(self, data):
        """
        Accept bytes from the host (e.g. alarm messages) and record them in
        the device's `received` list.

        Author: Bianca Dibden
        """
        if not self.is_open:
            raise SerialException("port is closed")
        if not self._alive() and self.dev.loss_mode == "error":
            raise SerialException("device disconnected")
        self.dev.received.append(bytes(data))
        return len(data)

    def close(self):
        """Close the connection. Author: Bianca Dibden"""
        self.is_open = False


def run_pty(args):
    """
    Serve simulated packets on a pseudo-terminal so another program (e.g. the
    GUI) can open it like the real Bluetooth port. Mac/Linux only.

    Author: Bianca Dibden
    """
    import tty
    master, slave = os.openpty()
    tty.setraw(slave)
    name = os.ttyname(slave)
    dev = SimulatedDevice(period=args.period, drop_prob=args.drop, corrupt_prob=args.corrupt,
                          swap_prob=args.swap, junk_prob=args.junk, seed=args.seed)
    print(f"Simulated ESP32 on port:  {name}")
    print(f"Open it with:  python3 comms.py {name}")
    print("Ctrl+C to stop.")
    k = 0
    try:
        while True:
            due = dev.packets_due()
            if due > k:
                data = dev.faulty_stream([dev.build(i) for i in range(k, due)])
                os.write(master, data)
                k = due
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Simulated ESP32 packet source.")
    ap.add_argument("--pty", action="store_true", help="serve packets on a pseudo-terminal")
    ap.add_argument("--period", type=float, default=1.0, help="seconds per packet")
    ap.add_argument("--drop", type=float, default=0.0, help="probability a packet is dropped")
    ap.add_argument("--corrupt", type=float, default=0.0, help="probability a packet is corrupted")
    ap.add_argument("--swap", type=float, default=0.0, help="probability of out-of-order delivery")
    ap.add_argument("--junk", type=float, default=0.0, help="probability of junk bytes before a packet")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.pty:
        run_pty(a)
    else:
        dev = SimulatedDevice()
        for k in range(3):
            print(dev.build(k).decode().strip())
