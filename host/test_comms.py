"""
Tests for the Bluetooth link (packet_format.py, comms.py), run without
hardware against the simulated ESP32 in packet_sim.py.

  1. Packet format: round trip, fixed length, and rejection of each kind of
     bad packet. If a C++ compiler is available, also checks that the ESP32's
     packet.cpp produces byte-identical packets.
  2. Sequence tracking: ok, wrap-around, lost, duplicate, out-of-order, restart.
  3. Normal operation: every packet received, HR correct, log format (Reqt 13).
  4. Fault injection: drops, corruption, junk bytes and reordering are
     detected, logged and survived (Reqt 18).
  5. Loss of comms (silent and error modes) and MCU reset: no-data alarm,
     automatic reconnection, alarm cleared (Reqts 14, 16).
  6. Real-time timing with the real parameters: alarm at 5 s, reconnection
     within 10 s of the ESP32 returning (Reqt 17), host latency (Reqt 12).

  8. Alarm limits sent to the ESP32 ($LIM message).

Run:  python3 test_comms.py          (about 40 s; add --quick to skip test 6)

Author: Bianca Dibden
Date: 15-Oct-2026
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import packet_format as pf
from comms import PPGLink, SequenceTracker, STATUS_CONNECTED
from packet_sim import SimulatedDevice

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = []
CTIME_RE = re.compile(r"^\w{3} \w{3} [ \d]\d \d{2}:\d{2}:\d{2} \d{4}: ")


def check(name, condition, detail=""):
    """
    Record and print a pass/fail result.

    Author: Bianca Dibden
    """
    RESULTS.append(bool(condition))
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def run_link(dev, seconds, **kw):
    """
    Run a PPGLink against a simulated device for a while.

    Returns
    -------
    link, records, events

    Author: Bianca Dibden
    """
    link = PPGLink("SIM", serial_factory=dev.open, **kw)
    link.start()
    records, events = [], []
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        records += link.get_records()
        events += link.get_events()
        time.sleep(0.02)
    link.stop()
    records += link.get_records()
    events += link.get_events()
    return link, records, events


def wait_for(link, cond, timeout, records, events):
    """
    Poll the link until cond() is true or timeout; collects records/events.

    Returns
    -------
    float or None
        Seconds taken, or None on timeout.

    Author: Bianca Dibden
    """
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        records += link.get_records()
        events += link.get_events()
        if cond():
            return time.monotonic() - t0
        time.sleep(0.01)
    return None


# Fast timing for most tests: 10x real time.
FAST = dict(retry_interval=0.1, reconnect_after=0.4, alarm_after=0.5)


def test_packet_format():
    """
    Test 1: packet construction and validation.

    Author: Bianca Dibden
    """
    print("\nTest 1 - packet format")
    samples = [1800 + 10 * i for i in range(50)]
    p = pf.build_packet(65535, 71.4, True, samples)
    q = pf.parse_packet(p)
    check("round trip and fixed length", len(p) == 268 and q.seq == 65535 and q.mcu_hr == 71.4
          and q.recording and q.samples == samples, f"{len(p)} bytes")

    lengths = {len(pf.build_packet(s, h, r, [v] * 50))
               for s in (0, 9, 65535) for h in (0, 7.25, 100, 999.9) for r in (0, 1) for v in (0, 4095)}
    check("every packet is 268 bytes", lengths == {268}, str(lengths))

    def reject(name, line, reason):
        try:
            pf.parse_packet(line)
            check(f"rejects {name}", False, "accepted")
        except pf.PacketError as e:
            check(f"rejects {name}", reason in str(e), str(e))

    bad = bytearray(p); bad[20] = ord("9") if bad[20] != ord("9") else ord("1")
    reject("single corrupted digit", bytes(bad), "checksum")
    reject("truncated packet", p[:150] + b"\n", "length")
    reject("missing start character", b"#" + p[1:], "start")

    def with_checksum(payload):
        return b"$" + payload + b"*" + f"{pf.checksum(payload):02X}".encode() + b"\n"
    payload = p[1:-4]
    reject("sample > 4095 (valid checksum)", with_checksum(payload[:-4] + b"5000"), "ADC")
    reject("bad recording flag", with_checksum(payload[:12] + b"7" + payload[13:]), "recording")
    reject("non-numeric field", with_checksum(payload[:20] + b"x" + payload[21:]), "non-numeric")

    # ESP32 C++ implementation must produce identical bytes
    cxx = shutil.which("g++") or shutil.which("clang++")
    src = os.path.join(HERE, "..", "esp32", "bt_packet_sender")
    if not cxx:
        print("  [SKIP] C++ compiler not found - ESP32 packet.cpp cross-check skipped")
        return
    with tempfile.TemporaryDirectory() as tmp:
        main = os.path.join(tmp, "main.cpp")
        exe = os.path.join(tmp, "pk")
        with open(main, "w") as f:
            f.write('#include <stdio.h>\n#include "packet.h"\nint main(){char b[PACKET_BUF_SIZE];uint16_t s[50];'
                    'float h[]={0,71.4f,99.96f,999.99f,-1};uint16_t q[]={0,1,65535,42,7};'
                    'for(int k=0;k<5;k++){for(int i=0;i<50;i++)s[i]=(k*517+i*83)%5000;'
                    'fwrite(b,1,buildPacket(b,sizeof b,q[k],h[k],k%2,s),stdout);}return 0;}')
        subprocess.run([cxx, "-o", exe, main, os.path.join(src, "packet.cpp"), "-I", src], check=True)
        out = subprocess.run([exe], capture_output=True, check=True).stdout
    lines = [l + b"\n" for l in out.split(b"\n")[:-1]]
    same = all(pf.build_packet(*(lambda r: (r.seq, r.mcu_hr, r.recording, r.samples))(pf.parse_packet(l))) == l
               for l in lines)
    check("ESP32 packet.cpp output is byte-identical to Python", len(lines) == 5 and same)


def test_sequence_tracker():
    """
    Test 2: sequence number classification.

    Author: Bianca Dibden
    """
    print("\nTest 2 - sequence tracking")
    t = SequenceTracker()
    got = [t.check(s) for s in (65534, 65535, 0, 3, 3, 2, 4, 900, 50)]
    want = [("first", 0), ("ok", 0), ("ok", 0), ("lost", 2), ("duplicate", 0),
            ("out_of_order", 0), ("ok", 0), ("restart", 0), ("restart", 0)]
    check("ok / wrap / lost / duplicate / out-of-order / restart", got == want, str(got))


def test_normal():
    """
    Test 3: clean link at 10x speed.

    Author: Bianca Dibden
    """
    print("\nTest 3 - normal operation (simulated ESP32, 10x speed)")
    dev = SimulatedDevice(period=0.1)
    link, recs, evs = run_link(dev, 4.0, **FAST)
    seqs = [r.seq for r in recs]
    check("received a continuous sequence", len(recs) >= 35 and seqs == list(range(seqs[0], seqs[0] + len(seqs))),
          f"{len(recs)} packets")
    check("no invalid or lost packets", link.stats["invalid"] == 0 and link.stats["lost"] == 0, str(link.stats))
    hrs = [r.host_hr for r in recs if r.host_hr is not None]
    check("host HR ~ 71 BPM from replayed pulse.csv", hrs and abs(sum(hrs) / len(hrs) - 71) < 3,
          f"mean {sum(hrs) / len(hrs):.1f} BPM over {len(hrs)} packets" if hrs else "none")
    check("MCU HR passed through", all(r.mcu_hr and 65 < r.mcu_hr < 76 for r in recs))
    check("log lines use '<time.ctime()>: <message>' (Reqt 13)", evs and all(CTIME_RE.match(e) for e in evs), evs[0] if evs else "")
    check("no false no-data alarm", not any("ALARM" in e for e in evs))


def test_faults():
    """
    Test 4: drops, corruption, junk and reordering.

    Author: Bianca Dibden
    """
    print("\nTest 4 - fault injection (10 % drop, 10 % corrupt, 10 % junk, 10 % reorder)")
    dev = SimulatedDevice(period=0.1, drop_prob=0.1, corrupt_prob=0.1, junk_prob=0.1, swap_prob=0.1, seed=3)
    link, recs, evs = run_link(dev, 8.0, **FAST)
    s = link.stats
    print(f"   produced {dev.packets_built}, stats {s}")
    check("corrupted packets rejected and logged", s["invalid"] > 0 and any("Invalid packet" in e for e in evs))
    check("lost packets detected and logged", s["lost"] > 0 and any("lost" in e for e in evs))
    check("out-of-order packets detected and dropped", s["out_of_order"] > 0)
    seqs = [r.seq for r in recs]
    check("accepted packets strictly increasing", all(b > a for a, b in zip(seqs, seqs[1:])))
    check("link stayed up (no reconnects needed)", s["connects"] == 1)
    check("no corrupted data reached the GUI",
          all(len(r.raw) == 50 and r.raw.min() >= 0 and r.raw.max() <= 4095 for r in recs))


def test_loss_and_recovery(mode):
    """
    Test 5: ESP32 disappears and returns.

    Author: Bianca Dibden
    """
    print(f"\nTest 5{'a' if mode == 'silent' else 'b'} - comms loss ({mode} link) and recovery, 10x speed")
    dev = SimulatedDevice(period=0.1, loss_mode=mode)
    link = PPGLink("SIM", serial_factory=dev.open, **FAST)
    recs, evs = [], []
    link.start()
    wait_for(link, lambda: len(recs) >= 10, 3, recs, evs)
    dev.power_off()
    t_alarm = wait_for(link, lambda: link.no_data_alarm, 2, recs, evs)
    check("no-data alarm raised", t_alarm is not None, f"after {t_alarm:.2f} s (threshold 0.5 s)" if t_alarm else "")
    check("loss logged", any("DISCONNECTED" in e and "lost" in e for e in evs))
    wait_for(link, lambda: False, 0.6, recs, evs)        # several failed reconnect attempts
    n_before = len(recs)
    dev.power_on()
    t_back = wait_for(link, lambda: len(recs) > n_before, 3, recs, evs)
    check("reconnected and data resumed", t_back is not None, f"{t_back:.2f} s after ESP32 returned" if t_back else "")
    wait_for(link, lambda: False, 0.1, recs, evs)
    check("alarm cleared and logged", not link.no_data_alarm and any("alarm cleared" in e for e in evs))
    check("status Connected", link.status == STATUS_CONNECTED)
    link.stop()

    if mode == "silent":
        print("\nTest 5c - ESP32 reset button (sequence restarts at 0)")
        dev = SimulatedDevice(period=0.1)
        link = PPGLink("SIM", serial_factory=dev.open, **FAST)
        recs, evs = [], []
        link.start()
        wait_for(link, lambda: len(recs) >= 15, 3, recs, evs)
        dev.reset(downtime=0.05)
        n = len(recs)
        t = wait_for(link, lambda: len(recs) > n + 3, 3, recs, evs)
        check("recovered after reset", t is not None and recs[-1].seq < 10,
              f"seq now {recs[-1].seq}, {t:.2f} s" if t else "")
        link.stop()


def test_real_timing():
    """
    Test 6: real-time parameters (1 packet/s, alarm 5 s, retry 1 s).

    Author: Bianca Dibden
    """
    print("\nTest 6 - real-time timing (1 packet/s, real thresholds) ~25 s")
    dev = SimulatedDevice(period=1.0)
    link = PPGLink("SIM", serial_factory=dev.open)
    recs, evs = [], []
    link.start()
    wait_for(link, lambda: len(recs) >= 4, 6, recs, evs)
    lat = [r.time - (dev.t_on + dev.period * (r.seq + 1)) for r in recs]
    lat = [x - 0 for x in lat]
    # dev.t_on is monotonic, r.time is wall clock - compare via processing latency instead
    proc = max(r.latency_s for r in recs)
    check("host processing per packet < 50 ms (Reqt 12 budget is 2 s)", proc < 0.05, f"max {proc * 1000:.1f} ms")
    dev.power_off()
    t_alarm = wait_for(link, lambda: link.no_data_alarm, 8, recs, evs)
    check("alarm raised ~5 s after last packet (Reqt 14)", t_alarm is not None and 3.5 <= t_alarm <= 6.0,
          f"{t_alarm:.1f} s after power-off" if t_alarm else "")
    wait_for(link, lambda: False, 2, recs, evs)
    n = len(recs)
    dev.power_on()
    t_back = wait_for(link, lambda: len(recs) > n, 12, recs, evs)
    check("data resumes within 10 s of ESP32 returning (Reqt 17)", t_back is not None and t_back < 10,
          f"{t_back:.1f} s (includes waiting for the first 1 s packet)" if t_back else "")
    link.stop()


def test_recording_flag():
    """
    Recording flag passes through and is logged.

    Author: Bianca Dibden
    """
    print("\nTest 7 - recording state")
    dev = SimulatedDevice(period=0.1)
    link = PPGLink("SIM", serial_factory=dev.open, **FAST)
    recs, evs = [], []
    link.start()
    wait_for(link, lambda: len(recs) >= 3, 2, recs, evs)
    dev.recording = True
    wait_for(link, lambda: recs and recs[-1].recording, 2, recs, evs)
    link.stop()
    check("recording state reaches the host and is logged",
          recs[-1].recording and any("Recording started" in e for e in evs))


def test_limits():
    """
    Test 8: HR alarm limits sent host -> ESP32 ($LIM message).

    Author: Bianca Dibden
    """
    print("\nTest 8 - alarm limits to the ESP32")
    msg = pf.build_limits(55, 99.5)
    check("limits message is 20 bytes and round-trips",
          msg == b"$LIM,055.0,099.5*" + msg[-3:] and len(msg) == pf.LIMITS_LEN
          and pf.parse_limits(msg) == (55.0, 99.5), msg.decode().strip())
    bad = 0
    for args in ((100, 60), (60, 60), (-1, 100), (50, 1000), (float("nan"), 100)):
        try:
            pf.build_limits(*args)
        except ValueError:
            bad += 1
    check("invalid limits are refused", bad == 5)
    try:
        pf.parse_limits(msg[:-3] + b"00\n")
        ok = False
    except pf.PacketError:
        ok = True
    check("corrupted limits message is rejected", ok)

    # Sent by the link, repeated, and re-sent after the ESP32 resets
    dev = SimulatedDevice(period=0.1)
    ports = []

    def factory(*a):
        ports.append(dev.open(*a))
        return ports[-1]
    link = PPGLink("SIM", serial_factory=factory, **FAST)
    recs, evs = [], []
    link.set_limits(65, 110)
    link.start()
    wait_for(link, lambda: ports and ports[0].limits == (65.0, 110.0), 2, recs, evs)
    check("limits reach the device after connecting", ports[0].limits == (65.0, 110.0))
    link.set_limits(40, 90)
    wait_for(link, lambda: ports[0].limits == (40.0, 90.0), 2, recs, evs)
    check("changed limits reach the device", ports[0].limits == (40.0, 90.0))
    link.stop()

    # C++ parser accepts what Python builds and rejects corruption
    cxx = shutil.which("g++") or shutil.which("clang++")
    src = os.path.join(HERE, "..", "A2_Lumi")
    if not cxx:
        print("  [SKIP] C++ compiler not found - ESP32 parseLimits cross-check skipped")
        return
    good = pf.build_limits(55, 99.5).rstrip(b"\n").decode()
    cases = [good, good[:-2] + "00", pf.build_limits(50, 100).rstrip(b"\n").decode(),
             "$LIM,100.0,060.0*00", good[:-1]]
    with tempfile.TemporaryDirectory() as tmp:
        main = os.path.join(tmp, "main.cpp")
        exe = os.path.join(tmp, "lim")
        body = "".join(f'{{const char*l="{c}";float a=0,b=0;bool r=parseLimits(l,strlen(l),&a,&b);'
                       f'printf("%d %.1f %.1f\\n",r,a,b);}}' for c in cases)
        with open(main, "w") as f:
            f.write('#include <stdio.h>\n#include <string.h>\n#include "packet.h"\nint main(){' + body + 'return 0;}')
        subprocess.run([cxx, "-o", exe, main, os.path.join(src, "packet.cpp"), "-I", src], check=True)
        out = subprocess.run([exe], capture_output=True, check=True).stdout.decode().split("\n")[:-1]
    check("ESP32 parseLimits agrees with Python",
          out == ["1 55.0 99.5", "0 0.0 0.0", "1 50.0 100.0", "0 0.0 0.0", "0 0.0 0.0"], str(out))


def test_blocking_connect():
    """
    Test 9: a reconnect attempt that blocks (a dead Bluetooth COM port can take
    10+ s to fail on Windows) must not delay the no-data alarm.

    Author: Bianca Dibden
    """
    print("\nTest 9 - alarm is not delayed by a blocking reconnect")
    dev = SimulatedDevice(period=0.1, loss_mode="silent")
    opens = []

    def slow_factory(port, baud, timeout):
        opens.append(time.monotonic())
        if len(opens) > 1:
            time.sleep(3.0)                 # dead port: open blocks, then fails
            raise OSError("could not open port")
        return dev.open(port, baud, timeout)

    link = PPGLink("SIM", serial_factory=slow_factory, **FAST)
    recs, evs = [], []
    link.start()
    wait_for(link, lambda: len(recs) >= 10, 3, recs, evs)
    dev.power_off()
    t_alarm = wait_for(link, lambda: link.no_data_alarm, 2.5, recs, evs)
    check("alarm raised while reconnect is blocked", t_alarm is not None and len(opens) > 1,
          f"after {t_alarm:.2f} s (threshold 0.5 s, open blocks 3 s)" if t_alarm else "")
    check("alarm log says device disconnected", any("NO-DATA ALARM" in e and "disconnected" in e for e in evs), str(evs[-2:]))
    link.stop()


if __name__ == "__main__":
    test_packet_format()
    test_sequence_tracker()
    test_normal()
    test_faults()
    test_loss_and_recovery("silent")
    test_loss_and_recovery("error")
    test_recording_flag()
    test_limits()
    test_blocking_connect()
    if "--quick" not in sys.argv:
        test_real_timing()
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed.")
