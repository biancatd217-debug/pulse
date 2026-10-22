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

  7. HR alarm limits (20-250 BPM) sent from the GUI to the ESP32, echoed back in
     every packet, re-sent until confirmed, restored after an ESP32 reset.

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
    check("limits echoed in packet", pf.parse_packet(pf.build_packet(1, 70, False, samples, 55, 120)).high_limit == 120)
    q = pf.parse_packet(p)
    check("round trip and fixed length", len(p) == 276 and q.seq == 65535 and q.mcu_hr == 71.4
          and q.recording and q.samples == samples
          and (q.low_limit, q.high_limit) == (40, 200), f"{len(p)} bytes")

    lengths = {len(pf.build_packet(s, h, r, [v] * 50))
               for s in (0, 9, 65535) for h in (0, 7.25, 100, 999.9) for r in (0, 1) for v in (0, 4095)}
    check("every packet is 276 bytes", lengths == {276}, str(lengths))

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
                    'fwrite(b,1,buildPacket(b,sizeof b,q[k],h[k],k%2,20+k*10,200+k*10,s),stdout);}return 0;}')
        subprocess.run([cxx, "-o", exe, main, os.path.join(src, "packet.cpp"), "-I", src], check=True)
        out = subprocess.run([exe], capture_output=True, check=True).stdout
    lines = [l + b"\n" for l in out.split(b"\n")[:-1]]
    same = all(pf.build_packet(*(lambda r: (r.seq, r.mcu_hr, r.recording, r.samples, r.low_limit, r.high_limit))(pf.parse_packet(l))) == l
               for l in lines)
    check("ESP32 packet.cpp output is byte-identical to Python", len(lines) == 5 and same)
    check_limits_command_cpp(cxx, src)


def check_limits_command_cpp(cxx, src):
    """
    The ESP32's parseLimitsCommand must accept exactly what the Python
    build_limits_command produces and reject the same bad commands.

    Author: Bianca Dibden
    """
    good = [pf.build_limits_command(lo, hi) for lo, hi in ((20, 21), (40, 200), (55, 120), (249, 250))]
    good_text = [c for c in good]
    bad = [b"$LIM,055,120*00\n",                                                            # wrong checksum
           b"$LIM,120,055*" + f"{pf.checksum(b'LIM,120,055'):02X}".encode() + b"\n",       # low > high
           b"$LIM,010,120*" + f"{pf.checksum(b'LIM,010,120'):02X}".encode() + b"\n",       # below range
           b"$LIM,050,300*" + f"{pf.checksum(b'LIM,050,300'):02X}".encode() + b"\n",       # above range
           b"$LIM,05x,120*" + f"{pf.checksum(b'LIM,05x,120'):02X}".encode() + b"\n",       # not digits
           b"$XYZ,055,120*" + f"{pf.checksum(b'XYZ,055,120'):02X}".encode() + b"\n"]       # wrong command
    lines = good_text + bad
    with tempfile.TemporaryDirectory() as tmp:
        main = os.path.join(tmp, "lim.cpp")
        exe = os.path.join(tmp, "lim")
        with open(main, "w") as f:
            f.write('#include <stdio.h>\n#include <string.h>\n#include "packet.h"\nint main(){char l[128];'
                    'while(fgets(l,sizeof l,stdin)){uint16_t a=0,b=0;'
                    'bool ok=parseLimitsCommand(l,strlen(l),&a,&b);'
                    'printf("%d %u %u\\n",ok,ok?a:0,ok?b:0);}return 0;}')
        subprocess.run([cxx, "-o", exe, main, os.path.join(src, "packet.cpp"), "-I", src], check=True)
        out = subprocess.run([exe], input=b"".join(lines), capture_output=True, check=True).stdout.decode().split("\n")
    results = [tuple(int(v) for v in l.split()) for l in out if l]
    check("ESP32 parseLimitsCommand accepts every valid command",
          results[:len(good)] == [(1, 20, 21), (1, 40, 200), (1, 55, 120), (1, 249, 250)], str(results[:len(good)]))
    check("ESP32 parseLimitsCommand rejects every bad command",
          all(r[0] == 0 for r in results[len(good):]) and len(results) == len(lines), str(results[len(good):]))


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
    check("loss logged", any("Connection lost" in e for e in evs))
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
    Test 8: HR alarm limits sent from the GUI to the ESP32.

    Author: Bianca Dibden
    """
    print("\nTest 8 - HR limits sent to the ESP32")
    c = pf.build_limits_command(55, 120)
    check("command format", c == b"$LIM,055,120*4B\n" and pf.parse_limits_command(c) == (55, 120), str(c))
    bad_rejected = True
    for lo, hi in ((10, 100), (100, 100), (120, 60), (50, 251)):
        try:
            pf.build_limits_command(lo, hi)
            bad_rejected = False
        except ValueError:
            pass
    check("out-of-range limits refused (20-250 BPM, low < high)", bad_rejected)
    flipped = bytearray(c); flipped[6] = ord("9")
    try:
        pf.parse_limits_command(bytes(flipped))
        check("corrupted command rejected", False)
    except pf.PacketError:
        check("corrupted command rejected", True)
    link = PPGLink("SIM")
    try:
        link.set_limits(300, 400)
        check("PPGLink.set_limits refuses invalid limits", False)
    except ValueError:
        check("PPGLink.set_limits refuses invalid limits", True)

    # Normal: ESP32 starts with the defaults, GUI sets 55-120.
    dev = SimulatedDevice(period=0.1)
    link = PPGLink("SIM", serial_factory=dev.open, **FAST)
    recs, evs = [], []
    link.set_limits(55, 120)
    link.start()
    t = wait_for(link, lambda: link.limits_confirmed, 4, recs, evs)
    wait_for(link, lambda: False, 0.3, recs, evs)      # let a following packet reach the record list
    check("limits applied on the ESP32 and confirmed", t is not None and (dev.low_limit, dev.high_limit) == (55, 120)
          and (recs[-1].device_low_limit, recs[-1].device_high_limit) == (55, 120), f"{t:.2f} s" if t else "")
    check("sent and confirmed events logged", any("sent to ESP32" in e for e in evs)
          and any("confirmed by ESP32" in e for e in evs))

    # Change while running.
    link.set_limits(50, 150)
    t = wait_for(link, lambda: link.limits_confirmed, 4, recs, evs)
    check("limits changed while running", t is not None and (dev.low_limit, dev.high_limit) == (50, 150))

    # Commands lost on the way are re-sent until the echo matches.
    dev.drop_commands = 2
    link.set_limits(45, 130)
    t = wait_for(link, lambda: link.limits_confirmed, 8, recs, evs)
    check("lost commands are re-sent until confirmed", t is not None and (dev.low_limit, dev.high_limit) == (45, 130),
          f"{t:.2f} s" if t else "")

    # ESP32 reset: it reverts to 40-200, the host notices and sets them again.
    dev.reset(downtime=0.05)
    check("ESP32 reboots with default limits", (dev.low_limit, dev.high_limit) == (40, 200))
    t = wait_for(link, lambda: link.limits_confirmed and dev.low_limit == 45, 8, recs, evs)
    check("limits restored after ESP32 reset", t is not None and (dev.low_limit, dev.high_limit) == (45, 130),
          f"{t:.2f} s" if t else "")
    link.stop()

    # Limits set before the ESP32 is reachable are delivered when it appears.
    dev = SimulatedDevice(period=0.1)
    dev.power_off()
    link = PPGLink("SIM", serial_factory=dev.open, **FAST)
    link.set_limits(60, 100)
    link.start()
    recs, evs = [], []
    wait_for(link, lambda: False, 0.5, recs, evs)
    dev.power_on()
    t = wait_for(link, lambda: link.limits_confirmed, 5, recs, evs)
    check("limits set while disconnected are sent on connect", t is not None and (dev.low_limit, dev.high_limit) == (60, 100))
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
    if "--quick" not in sys.argv:
        test_real_timing()
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed.")
