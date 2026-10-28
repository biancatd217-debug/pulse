"""
Host-side Bluetooth communications for the Pulse Sensor System (BMET2922).

PPGLink runs in a background thread and:
  - opens the ESP32's Bluetooth SPP serial port (Reqt 15),
  - frames and validates every packet (length, start char, checksum, ranges),
  - tracks sequence numbers to detect lost, duplicated or out-of-order packets
    and MCU restarts (Reqt 18),
  - raises a "no data" alarm if no valid packet arrives for 5 s, cleared by
    the next valid packet (Reqt 14),
  - on loss of comms keeps retrying to reconnect and resumes automatically
    when the ESP32 comes back (Reqts 16, 17),
  - passes each valid packet through the signal processing in ppg_dsp.py,
  - sends the HR alarm limits to the ESP32 (set_limits), repeated once a
    second, so its alarm/buzzer uses the limits chosen in the GUI,
  - hands the GUI one HostRecord per packet plus timestamped log events in
    the required "<time.ctime()>: <message>" format (Reqt 13).

The GUI never blocks: it simply calls get_records() / get_events() from its
event loop (e.g. after each window.read(timeout=100)).

    link = PPGLink("/dev/tty.BMET2922-G01")   # or "COM5", or "SIM" for the simulator
    link.start()
    ...
    for rec in link.get_records():  ...  # plot rec.raw / rec.filtered, show rec.host_hr
    for line in link.get_events():  ...  # append to the system log window
    link.set_limits(60, 100)            # ESP32 alarm limits (BPM)
    link.no_data_alarm                  # True while no packet for >= 5 s
    link.status                         # 'Connected' / 'Connecting' / 'Disconnected'
    ...
    link.stop()

Run directly for a console monitor (no GUI):
    python3 comms.py --list                  # show available serial ports
    python3 comms.py /dev/tty.BMET2922-G01   # monitor a real ESP32
    python3 comms.py SIM                     # monitor the built-in simulator

Author: Bianca Dibden
Date: 15-Oct-2026
"""

# ---------------------------------------------------------------------------
# HOW THE PIECES CONNECT (read this first)
#
#   ESP32 (A2_Lumi.ino)                          HOST (this folder)
#   -------------------                          ------------------
#   samples PPG @ 50 Hz
#   every 50 samples -> buildPacket()  ==BT==>   comms.py  PPGLink._read_once()
#     (A2_Lumi/packet.cpp)                         |  bytes -> lines ('\n' framing)
#                                                  v
#                                                PPGLink._handle_line()
#                                                  |  1. packet_format.parse_packet()  length/'$'/checksum/ranges
#                                                  |  2. SequenceTracker.check()       lost / duplicate / restart
#                                                  |  3. ppg_dsp.PPGProcessor.process_packet()
#                                                  |        filter -> beats -> HR -> FFT -> quality
#                                                  v
#                                                HostRecord  --queue-->  GUI (lp_gui.py)
#                                                                         calls get_records() / get_events()
#
#   GUI "Apply Limits" -> PPGLink.set_limits() -> packet_format.build_limits()
#   -> sent back to the ESP32 once a second ($LIM,...), read by readHostLimits().
#
# Threads: the GUI runs on the main thread; PPGLink._run() runs in a background
# thread and talks to the GUI only through two thread-safe queues (_records and
# _events), so the GUI never blocks on Bluetooth.
# ---------------------------------------------------------------------------

from __future__ import annotations

import argparse
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np

import packet_format as pf
from ppg_dsp import PPGProcessor

try:
    import serial
    from serial import SerialException
except ImportError:  # allows the simulator/tests to run without pyserial
    serial = None

    class SerialException(OSError):
        """Stand-in when pyserial is not installed. Author: Bianca Dibden"""

BAUD = 115200
READ_TIMEOUT_S = 0.1      # serial read timeout; keeps the thread responsive
RETRY_INTERVAL_S = 1.0    # wait between reconnection attempts (Reqt 17: < 10 s)
LIMITS_RESEND_S = 1.0     # repeat the limits message to the ESP32 this often
RECONNECT_AFTER_S = 4.0   # no bytes at all for this long on an open port -> close and reopen
ALARM_AFTER_S = 5.0       # no-packet alarm threshold (Reqt 14)
OUT_OF_ORDER_WINDOW = 5   # a seq up to this far behind = late packet; further = MCU restart
MAX_FORWARD_GAP = 300     # a jump forward bigger than this (5 min) is treated as a restart

STATUS_DISCONNECTED = "Disconnected"
STATUS_CONNECTING = "Connecting"
STATUS_CONNECTED = "Connected"


@dataclass
class HostRecord:
    """
    Everything the GUI needs for one second of data.

    Attributes
    ----------
    time : float
        Host time the packet was received (time.time()).
    seq : int
        Packet sequence number.
    mcu_hr : float or None
        Pulse rate calculated on the ESP32 (BPM); None if the MCU sent 0.
    recording : bool
        Recording state from the hardware button.
    raw : np.ndarray
        50 raw samples (for CSV recording / raw plot).
    filtered : np.ndarray
        50 band-pass filtered samples (for the waveform plot).
    threshold : np.ndarray
        Adaptive threshold for each filtered sample (optional plot overlay).
    host_hr : float or None
        Host-calculated pulse rate (BPM); None when the signal is not
        trustworthy. Display to 1 decimal place (Reqt 10).
    hr_fft : float or None
        FFT cross-check pulse rate (BPM) - useful for a Researcher tab.
    signal_ok : bool
        Signal validation verdict.
    signal_message : str
        e.g. 'Signal OK' or 'Signal poor: flat (no finger?)'.
    latency_s : float
        Host processing time for this packet (s). End-to-end latency is this
        plus up to 1 s of buffering on the MCU plus Bluetooth transmission.

    Author: Bianca Dibden
    """
    time: float
    seq: int
    mcu_hr: float | None
    recording: bool
    raw: np.ndarray
    filtered: np.ndarray
    threshold: np.ndarray
    host_hr: float | None
    hr_fft: float | None
    signal_ok: bool
    signal_message: str
    latency_s: float


class SequenceTracker:
    """
    Classifies each incoming sequence number relative to the last accepted one.

    Outcomes:
      'first'        - first packet since reset; accept
      'ok'           - exactly last + 1; accept
      'lost'         - jumped forward by 2..MAX_FORWARD_GAP; accept, n_lost > 0
      'duplicate'    - same as last; drop
      'out_of_order' - up to OUT_OF_ORDER_WINDOW behind; drop (data already shown)
      'restart'      - far behind or far ahead (MCU reset / long outage); accept

    Arithmetic is modulo 65536 so the wrap from 65535 to 0 is 'ok'.

    Author: Bianca Dibden
    """

    def __init__(self):
        """Author: Bianca Dibden"""
        self.last = None

    def reset(self):
        """
        Forget the last sequence number (call on reconnection).

        Author: Bianca Dibden
        """
        self.last = None

    def check(self, seq):
        """
        Classify seq and update the tracker if the packet should be accepted.

        Parameters
        ----------
        seq : int

        Returns
        -------
        outcome : str
        n_lost : int
            Number of packets missing (only for 'lost').

        Author: Bianca Dibden
        """
        if self.last is None:
            self.last = seq
            return "first", 0
        fwd = (seq - self.last) % pf.SEQ_MODULO
        if fwd == 1:
            outcome, n = "ok", 0
        elif fwd == 0:
            return "duplicate", 0
        elif fwd <= MAX_FORWARD_GAP:
            outcome, n = "lost", fwd - 1
        elif pf.SEQ_MODULO - fwd <= OUT_OF_ORDER_WINDOW:
            return "out_of_order", 0
        else:
            outcome, n = "restart", 0
        self.last = seq
        return outcome, n


class PPGLink:
    """
    Background Bluetooth SPP link to the ESP32 with validation, reconnection
    and signal processing. See the module docstring for usage.

    Author: Bianca Dibden
    """

    def __init__(self, port, baudrate=BAUD, serial_factory=None,
                 retry_interval=RETRY_INTERVAL_S, reconnect_after=RECONNECT_AFTER_S,
                 alarm_after=ALARM_AFTER_S, processor=None):
        """
        Parameters
        ----------
        port : str
            Serial port name: 'COM5' (Windows), '/dev/tty.<btName>' or
            '/dev/cu.<btName>' (Mac), or 'SIM' for the built-in simulator.
        baudrate : int
            Ignored by Bluetooth SPP but required by pyserial.
        serial_factory : callable, optional
            factory(port, baudrate, timeout) -> serial-like object. Used by the
            tests and simulator; defaults to serial.Serial.
        retry_interval, reconnect_after, alarm_after : float
            Timing parameters (s); see module constants.
        processor : PPGProcessor, optional
            Signal processing pipeline; a new one is created if omitted.

        Author: Bianca Dibden
        """
        self.port = port
        self.baudrate = baudrate
        if serial_factory is None and port in ("SIM", "SIM_QUALITY"):
            from packet_sim import SimulatedDevice
            if port == "SIM_QUALITY":
                # Scripted faults (no finger, saturation, noise, motion...) so the
                # signal-quality display can be seen working; see sim_quality_data.py.
                from sim_quality_data import make_quality_demo_data
                samples, rates = make_quality_demo_data()
                self.sim_device = SimulatedDevice(samples=samples, rates=rates)
            else:
                self.sim_device = SimulatedDevice()
            serial_factory = self.sim_device.open
        self.serial_factory = serial_factory or self._open_real_port
        self.retry_interval = retry_interval
        self.reconnect_after = reconnect_after
        self.alarm_after = alarm_after
        self.processor = processor or PPGProcessor()

        self.status = STATUS_DISCONNECTED
        self.no_data_alarm = False
        self.stats = {"valid": 0, "invalid": 0, "lost": 0, "out_of_order": 0,
                      "duplicate": 0, "restarts": 0, "connects": 0}
        self.last_record = None

        self._records = queue.Queue()
        self._events = queue.Queue()
        self._seq = SequenceTracker()
        self._ser = None
        self._buf = bytearray()
        self._stop = threading.Event()
        self._thread = None
        self._last_valid = time.monotonic()
        self._last_rx = time.monotonic()     # last time ANY bytes arrived
        self._connected_at = None
        self._valid_since_connect = 0
        self._attempts = 0

        # HR alarm limits to send to the ESP32 (None until the GUI sets them).
        self._limits_msg = None
        self._limits_tx_time = 0.0
        self._limits_lock = threading.Lock()

    # ----------------------------------------------------------------- public
    def start(self):
        """
        Start the background reader thread.

        Author: Bianca Dibden
        """
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._last_valid = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="PPGLink", daemon=True)
        self._thread.start()

    def stop(self):
        """
        Stop the thread and close the port. Call when the GUI exits.

        Author: Bianca Dibden
        """
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        self._close()
        self.status = STATUS_DISCONNECTED

    def get_records(self):
        """
        Return all HostRecords received since the last call (non-blocking).

        Returns
        -------
        list of HostRecord

        Author: Bianca Dibden
        """
        return self._drain(self._records)

    def get_events(self):
        """
        Return all log lines generated since the last call (non-blocking),
        each formatted '<time.ctime()>: <message>' (Reqt 13).

        Returns
        -------
        list of str

        Author: Bianca Dibden
        """
        return self._drain(self._events)

    def set_limits(self, low, high):
        """
        Set the HR alarm limits the ESP32 should use. Non-blocking: the link
        thread sends them at once and repeats them every LIMITS_RESEND_S, so a
        reconnection or ESP32 reset picks them up again.

        Parameters
        ----------
        low, high : float
            Limits in BPM, 0 <= low < high <= 999.9.

        Raises
        ------
        ValueError
            If the limits are invalid (nothing is sent).

        Author: Bianca Dibden
        """
        msg = pf.build_limits(low, high)
        with self._limits_lock:
            self._limits_msg = msg
            self._limits_tx_time = 0.0      # send immediately

    @property
    def seconds_since_packet(self):
        """
        Seconds since the last valid packet (or since start).

        Author: Bianca Dibden
        """
        return time.monotonic() - self._last_valid

    # --------------------------------------------------------------- internal
    @staticmethod
    def _drain(q):
        """Empty a queue into a list. Author: Bianca Dibden"""
        items = []
        while True:
            try:
                items.append(q.get_nowait())
            except queue.Empty:
                return items

    def _log(self, message):
        """
        Queue a timestamped event for the GUI system log.

        Author: Bianca Dibden
        """
        self._events.put(f"{time.ctime()}: {message}")

    def _open_real_port(self, port, baudrate, timeout):
        """
        Open a real serial port with pyserial.

        Author: Bianca Dibden
        """
        if serial is None:
            raise SerialException("pyserial is not installed (pip3 install pyserial)")
        return serial.Serial(port=port, baudrate=baudrate, timeout=timeout, write_timeout=1)

    def _run(self):
        """
        Thread main loop: connect / read / check timeouts until stopped.

        Author: Bianca Dibden
        """
        while not self._stop.is_set():
            # State A - not connected: try to open the port, wait, try again
            # (this is the "keep retrying" behaviour of Reqts 16-17).
            if self._ser is None:
                if not self._try_connect():
                    self._check_alarm()
                    self._stop.wait(self.retry_interval)
                    continue
            # State B - connected: one pass of the loop does four jobs.
            self._read_once()       # 1. read bytes, validate + process any full packets
            self._send_limits()     # 2. resend the GUI's HR limits to the ESP32 if due
            self._check_alarm()     # 3. no valid packet for 5 s -> raise GUI alarm
            self._check_silence()   # 4. port open but no bytes at all -> force a reconnect

    def _try_connect(self):
        """
        Attempt to open the port once.

        Returns
        -------
        bool
            True if the port opened.

        Author: Bianca Dibden
        """
        self.status = STATUS_CONNECTING
        self._attempts += 1
        try:
            ser = self.serial_factory(self.port, self.baudrate, READ_TIMEOUT_S)
            try:
                ser.reset_input_buffer()
            except (AttributeError, OSError, SerialException):
                pass
        except (SerialException, OSError, ValueError) as e:
            if self._attempts == 1 or self._attempts % 10 == 0:
                self._log(f"Connecting to {self.port}... (attempt {self._attempts}: {short_error(e)})")
            self.status = STATUS_DISCONNECTED   # attempt failed: no connection until one succeeds
            return False

        self._ser = ser
        self._buf.clear()
        self._seq.reset()
        self.processor.reset()
        self._connected_at = time.monotonic()
        self._last_rx = self._connected_at
        self._valid_since_connect = 0
        self.status = STATUS_CONNECTED
        self.stats["connects"] += 1
        what = "Connected" if self.stats["connects"] == 1 else "Reconnected"
        self._log(f"{what} to {self.port}" + (f" after {self._attempts} attempt(s)" if self._attempts > 1 else ""))
        self._attempts = 0
        return True

    def _close(self):
        """
        Close the port, ignoring errors.

        Author: Bianca Dibden
        """
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
        self._ser = None

    def _send_limits(self):
        """
        Send the limits message if one is set and it is due.

        Author: Bianca Dibden
        """
        if self._ser is None:
            return
        now = time.monotonic()
        with self._limits_lock:
            msg = self._limits_msg
            if msg is None or now - self._limits_tx_time < LIMITS_RESEND_S:
                return
            self._limits_tx_time = now
        try:
            self._ser.write(msg)
        except (SerialException, OSError) as e:
            self._lost(short_error(e))

    def _lost(self, reason):
        """
        Handle a broken link: log, close, and fall back to reconnecting.
        Status is Disconnected until the next attempt opens the port.

        Author: Bianca Dibden
        """
        self._log(f"Connection lost ({reason})")
        self._close()
        self.status = STATUS_DISCONNECTED

    def _read_once(self):
        """
        Read whatever bytes are available (blocking at most READ_TIMEOUT_S),
        split them into lines, and handle each complete line.

        Author: Bianca Dibden
        """
        try:
            n = getattr(self._ser, "in_waiting", 0) or 1
            data = self._ser.read(n)
        except (SerialException, OSError) as e:
            self._lost(short_error(e))
            return
        if not data:
            return
        self._last_rx = time.monotonic()
        self._buf.extend(data)

        while True:
            nl = self._buf.find(b"\n")
            if nl < 0:
                break
            line = bytes(self._buf[:nl + 1])
            del self._buf[:nl + 1]
            self._handle_line(line)

        # Never let garbage without newlines grow the buffer without bound:
        # keep only from the last start character.
        if len(self._buf) > 2 * pf.PACKET_LEN:
            start = self._buf.rfind(pf.START)
            del self._buf[:start if start > 0 else len(self._buf)]

    def _handle_line(self, line):
        """
        Validate one line and, if valid, process it into a HostRecord.

        Author: Bianca Dibden
        """
        # --- Step 1: framing. Bluetooth is a byte stream, so a "line" may start
        # with leftovers of a previous packet. Keep only from the last '$'.
        start = line.rfind(pf.START)
        if start > 0:
            line = line[start:]
        if line.strip() == b"":
            return
        # --- Step 2: validation. parse_packet checks length, '$', checksum, field
        # count and value ranges. Anything corrupt is logged and discarded.
        try:
            pkt = pf.parse_packet(line)
        except pf.PacketError as e:
            # The first line after connecting is usually a partial packet; ignore it quietly.
            if self._valid_since_connect == 0 and not line.startswith(pf.START):
                return
            self.stats["invalid"] += 1
            self._log(f"Invalid packet discarded ({e})")
            return

        # --- Step 3: sequence check. Drop repeats / late packets; if packets were
        # lost or the ESP32 restarted, reset the DSP because the 50 Hz stream is
        # no longer continuous (filter and beat timing would be wrong).
        outcome, n_lost = self._seq.check(pkt.seq)
        if outcome in ("duplicate", "out_of_order"):
            self.stats[outcome] += 1
            self._log(f"Packet {pkt.seq} {outcome.replace('_', ' ')} - dropped")
            return
        if outcome == "lost":
            self.stats["lost"] += n_lost
            self._log(f"{n_lost} packet(s) lost before packet {pkt.seq}")
            self.processor.reset()     # data is no longer continuous
        elif outcome == "restart":
            self.stats["restarts"] += 1
            self._log(f"Sequence restarted at {pkt.seq} (ESP32 reset?)")
            self.processor.reset()

        # --- Step 4: signal processing (ppg_dsp.py). Only the 50 RAW samples go in;
        # the MCU's own HR (pkt.mcu_hr) is NOT used by the host DSP, it is just
        # carried along so the GUI can show MCU HR next to host HR.
        t0 = time.perf_counter()
        result = self.processor.process_packet(pkt.samples)
        # --- Step 5: bundle everything the GUI needs into one HostRecord and hand
        # it over through the queue (the GUI picks it up with get_records()).
        rec = HostRecord(
            time=time.time(), seq=pkt.seq,
            mcu_hr=pkt.mcu_hr if pkt.mcu_hr > 0 else None,
            recording=pkt.recording,
            raw=result.raw, filtered=result.filtered, threshold=result.threshold,
            host_hr=result.host_hr, hr_fft=result.hr_fft,
            signal_ok=result.quality.ok, signal_message=result.quality.message(),
            latency_s=time.perf_counter() - t0,
        )
        if self.last_record is not None and rec.recording != self.last_record.recording:
            self._log("Recording started" if rec.recording else "Recording stopped")
        self.last_record = rec
        self._records.put(rec)

        self.stats["valid"] += 1
        self._valid_since_connect += 1
        self._last_valid = time.monotonic()
        if self.no_data_alarm:
            self.no_data_alarm = False
            self._log("Data resumed - no-data alarm cleared")

    def _check_alarm(self):
        """
        Raise the no-data alarm after alarm_after seconds without a valid packet.

        Author: Bianca Dibden
        """
        if not self.no_data_alarm and self.seconds_since_packet >= self.alarm_after:
            self.no_data_alarm = True
            self._log(f"ALARM: no data for {self.alarm_after:.0f} s")

    def _check_silence(self):
        """
        If the port is open but no bytes at all arrive for reconnect_after
        seconds, assume the link is dead (e.g. ESP32 reset - the virtual port
        stays open but nothing flows) and close it so the reconnect loop takes
        over. Bytes that fail validation still prove the link is alive, so a
        burst of corrupted packets raises the no-data alarm but does not
        trigger a needless reconnect.

        Author: Bianca Dibden
        """
        if self._ser is None:
            return
        if time.monotonic() - self._last_rx >= self.reconnect_after:
            self._lost(f"no data for {self.reconnect_after:.0f} s")


def short_error(e):
    """
    One-line description of an exception for the log.

    Author: Bianca Dibden
    """
    text = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
    return text[:80]


def _console_monitor(port):
    """
    Print records and events to the terminal (testing without the GUI).

    Author: Bianca Dibden
    """
    link = PPGLink(port)
    link.start()
    print(f"Monitoring {port} - Ctrl+C to stop")
    print(" seq    MCU HR   host HR   FFT HR   rec  signal")
    try:
        while True:
            for line in link.get_events():
                print("  LOG  " + line)
            for r in link.get_records():
                f = lambda v: "   -  " if v is None else f"{v:6.1f}"
                print(f"{r.seq:05d}  {f(r.mcu_hr)}   {f(r.host_hr)}   {f(r.hr_fft)}    "
                      f"{int(r.recording)}   {r.signal_message}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        link.stop()
        print("Stats:", link.stats)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Monitor the ESP32 Bluetooth link.")
    ap.add_argument("port", nargs="?", help="serial port, e.g. COM5, /dev/tty.BMET2922-G01, or SIM")
    ap.add_argument("--list", action="store_true", help="list serial ports and exit")
    args = ap.parse_args()
    if args.list or not args.port:
        if serial is None:
            print("pyserial not installed: pip3 install pyserial")
        else:
            from serial.tools import list_ports
            for p in list_ports.comports():
                print(f"{p.device:35s} {p.description}")
        if not args.port:
            raise SystemExit(0)
    _console_monitor(args.port)
