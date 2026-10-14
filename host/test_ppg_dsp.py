"""
Offline tests for ppg_dsp.py, run without any hardware.

1. pulse.csv (course data, 12.8 s @ 50 Hz) fed in 50-sample "packets":
   host HR must match the reference pulse rate in column 2.
2. Robustness to low-frequency content (Reqt 19): pulse.csv-like synthetic PPG
   with respiration baseline wander, slow drift, a contact-pressure step and
   noise. Compares a fixed threshold (Lab 3 style) with the band-pass +
   adaptive-threshold pipeline.
3. HR tracking: synthetic PPG whose rate ramps 60 -> 110 BPM.
4. Signal validation: no finger (flat), saturated, and pure noise must all be
   flagged as not ok, with host_hr = None.
5. Offline analysis of a whole recording (summary-report path).

Figures are written to Code/figures for the report.
Run:  python3 test_ppg_dsp.py

Author: Bianca Dibden
Date: 15-Oct-2026
"""

import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ppg_dsp as dsp

HERE = os.path.dirname(os.path.abspath(__file__))
PULSE_CSV = os.path.join(HERE, "..", "data", "pulse.csv")
FIG_DIR = os.path.join(HERE, "..", "figures")
FS = dsp.FS
RESULTS = []


def check(name, condition, detail=""):
    """
    Record and print a pass/fail result.

    Author: Bianca Dibden
    """
    RESULTS.append(bool(condition))
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def load_pulse_csv(path=PULSE_CSV):
    """
    Load the course pulse.csv file.

    Returns
    -------
    raw : np.ndarray
        Raw ADC samples.
    ref_bpm : np.ndarray
        Reference pulse rate, converted from Hz to BPM.

    Author: Bianca Dibden
    """
    raw, ref = [], []
    with open(path) as f:
        for row in csv.reader(f):
            raw.append(int(row[0]))
            ref.append(float(row[1]) * 60.0)
    return np.array(raw, float), np.array(ref)


def synthetic_ppg(duration_s, hr_bpm, fs=FS, amplitude=300.0, seed=0, return_beats=False):
    """
    Generate a realistic-looking PPG: per beat, a systolic Gaussian peak plus a
    smaller, later dicrotic bump. hr_bpm may be a constant or a function of time.

    Returns
    -------
    np.ndarray
        Signal (ADC counts) centred on 2000.
    np.ndarray (only if return_beats)
        True systolic peak times (s).

    Author: Bianca Dibden
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    x = np.zeros(n)
    beat_t = 0.3
    beats = []
    while beat_t < duration_s + 1:
        beats.append(beat_t)
        hr = hr_bpm(beat_t) if callable(hr_bpm) else hr_bpm
        period = 60.0 / hr * (1 + 0.02 * rng.standard_normal())
        x += amplitude * np.exp(-0.5 * ((t - beat_t) / (0.10 * period)) ** 2)
        x += 0.35 * amplitude * np.exp(-0.5 * ((t - beat_t - 0.35 * period) / (0.08 * period)) ** 2)
        beat_t += period
    x = 2000 + x - x.mean()
    if return_beats:
        b = np.array(beats)
        return x, b[b < duration_s]
    return x


def add_low_freq_artefacts(x, fs=FS, seed=1):
    """
    Add the low-frequency problems Reqt 19 is about: respiration wander
    (0.25 Hz), slow drift, a sudden contact-pressure step, and sensor noise.

    Author: Bianca Dibden
    """
    rng = np.random.default_rng(seed)
    t = np.arange(len(x)) / fs
    wander = 250 * np.sin(2 * np.pi * 0.25 * t)
    drift = 15 * t
    step = np.where(t > t[-1] * 0.55, -400.0, 0.0)
    noise = 15 * rng.standard_normal(len(x))
    return x + wander + drift + step + noise


def run_stream(x, processor=None):
    """
    Feed a signal through PPGProcessor in 50-sample packets.

    Returns
    -------
    dict of np.ndarray / lists
        filtered, threshold, beat_times, and per-packet t, host_hr, hr_peak,
        hr_fft, ok, reasons.

    Author: Bianca Dibden
    """
    proc = processor or dsp.PPGProcessor()
    out = {k: [] for k in ("filtered", "threshold", "beats", "t", "host_hr",
                           "hr_peak", "hr_fft", "ok", "reasons")}
    for k in range(len(x) // dsp.SAMPLES_PER_PACKET):
        pkt = x[k * 50:(k + 1) * 50]
        r = proc.process_packet(pkt)
        out["filtered"].extend(r.filtered)
        out["threshold"].extend(r.threshold)
        out["beats"].extend(r.beat_times)
        out["t"].append(k + 1)
        out["host_hr"].append(r.host_hr)
        out["hr_peak"].append(r.hr_peak)
        out["hr_fft"].append(r.hr_fft)
        out["ok"].append(r.quality.ok)
        out["reasons"].append(r.quality.reasons)
    return out


def fmt(v):
    """Format an optional BPM value. Author: Bianca Dibden"""
    return "  -  " if v is None else f"{v:5.1f}"


# ---------------------------------------------------------------------------
def test_pulse_csv():
    """
    Test 1: course data, packet by packet.

    Author: Bianca Dibden
    """
    print("\nTest 1 - pulse.csv streamed as 1 s packets")
    raw, ref = load_pulse_csv()
    res = run_stream(raw)
    print("   t(s)  host_HR  peak_HR  FFT_HR  quality")
    for t, h, p, f, r in zip(res["t"], res["host_hr"], res["hr_peak"], res["hr_fft"], res["reasons"]):
        print(f"   {t:4d}   {fmt(h)}    {fmt(p)}   {fmt(f)}  {'ok' if not r else ','.join(r)}")
    valid = [h for h in res["host_hr"] if h is not None]
    ref_mean = ref.mean()
    check("host HR produced after warm-up", len(valid) >= 5, f"{len(valid)} of {len(res['t'])} packets")
    if valid:
        err = abs(np.mean(valid) - ref_mean)
        check("host HR within 3 BPM of reference", err <= 3,
              f"host {np.mean(valid):.1f} vs ref {ref_mean:.1f} BPM")

    # Figure: raw vs filtered with beats and adaptive threshold
    t = np.arange(len(res["filtered"])) / FS
    fig, ax = plt.subplots(2, 1, figsize=(8, 4.5), sharex=True)
    ax[0].plot(np.arange(len(raw)) / FS, raw, lw=1)
    ax[0].set_ylabel("Raw (ADC counts)")
    ax[0].set_title("pulse.csv processed in 1 s packets")
    ax[1].plot(t, res["filtered"], lw=1, label="Band-pass 0.5-5 Hz")
    ax[1].plot(t, res["threshold"], "--", lw=1, label="Adaptive threshold")
    bt = np.array(res["beats"])
    ax[1].plot(bt, np.interp(bt, t, res["filtered"]), "rv", label="Detected beats")
    ax[1].set_ylabel("Filtered (counts)")
    ax[1].set_xlabel("Time (s)")
    ax[1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_pulse_csv_pipeline.png"), dpi=200)
    plt.close(fig)


def test_low_frequency_robustness():
    """
    Test 2 (Reqt 19): fixed threshold vs band-pass + adaptive threshold on a
    signal with baseline wander, drift, a pressure step and noise.

    Author: Bianca Dibden
    """
    print("\nTest 2 - low-frequency artefacts (Reqt 19), true HR = 72 BPM")
    true_hr = 72.0
    clean, true_beats = synthetic_ppg(40, true_hr, return_beats=True)
    dirty = add_low_freq_artefacts(clean)

    # Lab 3 style: one fixed threshold at the mean of the first 3 s, rising edges
    fixed_thr = dirty[:3 * FS].mean() + 0.2 * dirty[:3 * FS].std()
    above = dirty > fixed_thr
    edges = np.flatnonzero(~above[:-1] & above[1:]) + 1
    fixed_ibi = np.diff(edges) / FS
    fixed_hr = 60 / np.median(fixed_ibi) if len(fixed_ibi) else float("nan")
    expected_beats = len(true_beats)
    expected_after = int(np.sum(true_beats > 3))

    res = run_stream(dirty)
    valid = [h for h in res["host_hr"][5:] if h is not None]
    pipe_hr = np.median(valid) if valid else float("nan")
    print(f"   fixed threshold: {len(edges)} beats (expected ~{expected_beats:.0f}), HR {fixed_hr:.1f} BPM")
    print(f"   pipeline       : {len(res['beats'])} beats, median host HR {pipe_hr:.1f} BPM,"
          f" valid in {len(valid)}/{len(res['host_hr'][5:])} packets after warm-up")
    check("pipeline HR within 3 BPM of truth despite drift", abs(pipe_hr - true_hr) <= 3,
          f"{pipe_hr:.1f} BPM")
    found_after = int(np.sum(np.array(res["beats"]) > 3))
    check("pipeline finds every beat after 3 s warm-up (within 1)",
          abs(found_after - expected_after) <= 1, f"{found_after} found vs {expected_after} true")
    check("fixed threshold miscounts beats (shows the problem)", abs(len(edges) - expected_beats) > 2)

    t = np.arange(len(dirty)) / FS
    tf = np.arange(len(res["filtered"])) / FS
    fig, ax = plt.subplots(3, 1, figsize=(8, 6.5), sharex=True)
    ax[0].plot(t, dirty, lw=0.8)
    ax[0].axhline(fixed_thr, color="r", ls="--", lw=1, label="Fixed threshold")
    ax[0].plot(edges / FS, dirty[edges], "rv", ms=4, label=f"Fixed: {len(edges)} beats")
    ax[0].set_ylabel("Raw + artefacts")
    ax[0].legend(loc="upper right", fontsize=8)
    ax[0].set_title(f"Baseline wander, drift and pressure step (true HR {true_hr:.0f} BPM)")
    ax[1].plot(tf, res["filtered"], lw=0.8, label="Band-pass")
    ax[1].plot(tf, res["threshold"], "--", lw=0.8, label="Adaptive threshold")
    bt = np.array(res["beats"])
    ax[1].plot(bt, np.interp(bt, tf, res["filtered"]), "rv", ms=4, label=f"Pipeline: {len(bt)} beats")
    ax[1].set_ylabel("Filtered")
    ax[1].legend(loc="upper right", fontsize=8)
    hh = [np.nan if h is None else h for h in res["host_hr"]]
    ff = [np.nan if h is None else h for h in res["hr_fft"]]
    ax[2].plot(res["t"], hh, "o-", ms=3, label="Host HR (adaptive threshold)")
    ax[2].plot(res["t"], ff, "s--", ms=3, label="FFT HR (8 s window)")
    ax[2].axhline(true_hr, color="k", lw=0.8, ls=":", label="True HR")
    ax[2].set_ylim(40, 120)
    ax[2].set_ylabel("HR (BPM)")
    ax[2].set_xlabel("Time (s)")
    ax[2].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_drift_robustness.png"), dpi=200)
    plt.close(fig)


def test_hr_tracking():
    """
    Test 3: HR ramps 60 -> 110 BPM over 60 s; host HR must follow.

    Author: Bianca Dibden
    """
    print("\nTest 3 - tracking a changing HR (60 -> 110 BPM over 60 s)")
    ramp = lambda t: 60 + 50 * min(max(t, 0), 60) / 60
    x = add_low_freq_artefacts(synthetic_ppg(60, ramp, seed=3), seed=4)
    res = run_stream(x)
    errs = []
    for t, h in zip(res["t"], res["host_hr"]):
        if h is not None and t > 5:
            errs.append(abs(h - ramp(t - 2)))   # median of last ~6 beats lags ~2-3 s
    mae = np.mean(errs) if errs else float("inf")
    check("mean tracking error < 5 BPM", mae < 5, f"MAE {mae:.1f} BPM over {len(errs)} packets")


def test_signal_validation():
    """
    Test 4: bad signals must be flagged and must not produce a host HR.

    Author: Bianca Dibden
    """
    print("\nTest 4 - signal validation")
    rng = np.random.default_rng(5)
    cases = {
        "no finger (flat + small noise)": (1800 + 3 * rng.standard_normal(15 * FS), "flat"),
        "saturated (stuck at 4095)": (np.clip(synthetic_ppg(15, 70, amplitude=3000) + 2000, 0, 4095), "saturated"),
        "random noise (no pulse)": (2000 + 200 * rng.standard_normal(15 * FS), None),
    }
    for name, (x, expected) in cases.items():
        res = run_stream(x)
        after = list(zip(res["ok"][5:], res["host_hr"][5:], res["reasons"][5:]))
        rejected = sum(1 for ok, h, _ in after if not ok and h is None)
        reasons = sorted({r for _, _, rs in after for r in rs})
        frac = rejected / len(after)
        ok = frac >= 0.8 and (expected is None or expected in reasons)
        check(f"{name}: rejected", ok, f"{rejected}/{len(after)} packets, reasons {reasons}")

    # A good signal should mostly pass
    res = run_stream(synthetic_ppg(20, 75, seed=6))
    frac_ok = np.mean(res["ok"][4:])
    check("clean 75 BPM signal: accepted", frac_ok >= 0.9, f"{frac_ok:.0%} of packets ok")


def test_offline():
    """
    Test 5: offline whole-record analysis (for the summary report).

    Author: Bianca Dibden
    """
    print("\nTest 5 - offline analysis (zero-phase)")
    raw, ref = load_pulse_csv()
    a = dsp.analyse_recording(raw)
    print(f"   pulse.csv: {len(a['beat_times'])} beats, mean HR {a['mean_hr']:.1f} BPM, "
          f"FFT HR {a['hr_fft']:.1f} BPM, SDNN {a['sdnn_ms']:.0f} ms, RMSSD {a['rmssd_ms']:.0f} ms")
    check("offline mean HR within 3 BPM of reference", abs(a["mean_hr"] - ref.mean()) <= 3)
    check("offline FFT HR within 3 BPM of reference", abs(a["hr_fft"] - ref.mean()) <= 3)


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    test_pulse_csv()
    test_low_frequency_robustness()
    test_hr_tracking()
    test_signal_validation()
    test_offline()
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed. Figures in {FIG_DIR}")
