"""
Scripted "bad signal" test data for the simulated ESP32 (BMET2922 Major Project).

pulse.csv is a clean recording, so the signal-quality checks in ppg_dsp.py never
trigger on it. This module builds a second, synthetic data set that steps through
a series of faults so the GUI's quality display can be seen working:

    python3 lp_gui.py SIM_QUALITY          # GUI on the faulty data
    python3 comms.py SIM_QUALITY           # console view
    python3 sim_quality_data.py            # print the timeline

The sequence loops. Times are seconds from the start of the data (one packet = 1 s).
Stretches are long because the quality check looks back over the last 8 s: a
verdict follows a change by roughly 5-8 s, and after a fault the message can
linger for a few seconds while clean data refills the window.
SCHEDULE below lists: name, kind, duration (s), BPM, expected host message.

Author: Bianca Dibden
Date: 27-Oct-2026
"""

from __future__ import annotations

import numpy as np

FS = 50                  # samples per second (same as the ESP32)
BASELINE = 2000          # typical resting ADC level of the sensor
AMPLITUDE = 350          # pulse height in ADC counts

# (name, kind, seconds, bpm, expected host message)
SCHEDULE = [
    ("Clean pulse (warm-up)",        "clean",     14, 72,  "acquiring signal, then Signal OK"),
    ("No finger on sensor",          "flat",      12, 0,   "Signal poor: flat (no finger?)"),
    ("Finger back on",               "clean",     14, 75,  "Signal OK (recovers)"),
    ("Sensor saturated",             "saturated", 14, 70,  "Signal poor: saturated"),
    ("Clean again",                  "clean",     14, 70,  "Signal OK"),
    ("Random noise, no pulse",       "noise",     14, 0,   "pulse methods disagree / irregular / no clear rhythm"),
    ("Clean again",                  "clean",     14, 80,  "Signal OK"),
    ("Movement artefact",            "motion",    18, 72,  "disagree / irregular beat intervals"),
    ("Clean again",                  "clean",     14, 72,  "Signal OK"),
    ("Irregular rhythm",             "irregular", 18, 75,  "irregular beat intervals"),
    ("Clean pulse",                  "clean",     14, 68,  "Signal OK"),
]


def _pulse_train(n, bpm, rng, jitter=0.0):
    """
    Synthetic PPG pulse train: a systolic peak plus a smaller dicrotic bump.

    Parameters
    ----------
    n : int
        Number of samples.
    bpm : float
        Mean heart rate.
    rng : numpy.random.Generator
    jitter : float
        Random variation of each beat interval (0.3 = +/-30 %).

    Returns
    -------
    np.ndarray
        Waveform, 0..~1.

    Author: Bianca Dibden
    """
    t = np.arange(n) / FS
    y = np.zeros(n)
    beat = rng.uniform(0.0, 60.0 / bpm)
    while beat < t[-1] + 1.0:
        y += np.exp(-((t - beat - 0.12) ** 2) / (2 * 0.04 ** 2))
        y += 0.35 * np.exp(-((t - beat - 0.38) ** 2) / (2 * 0.06 ** 2))
        beat += (60.0 / bpm) * (1.0 + rng.uniform(-jitter, jitter))
    return y


def _segment(kind, seconds, bpm, rng):
    """
    Build one stretch of ADC samples of the given kind.

    Parameters
    ----------
    kind : str
        'clean', 'flat', 'saturated', 'noise', 'motion' or 'irregular'.
    seconds : int
    bpm : float
    rng : numpy.random.Generator

    Returns
    -------
    samples : np.ndarray
        Float ADC values (not yet clipped/rounded).
    mcu_hr : float
        What the ESP32 would plausibly report for this stretch (BPM, 0 = none).

    Author: Bianca Dibden
    """
    n = seconds * FS
    t = np.arange(n) / FS
    noise = rng.normal(0, 6, n)

    if kind == "clean":
        return BASELINE + AMPLITUDE * _pulse_train(n, bpm, rng, 0.03) + noise, bpm
    if kind == "flat":                      # no finger: nothing but sensor noise
        return BASELINE + rng.normal(0, 2, n), 0.0
    if kind == "saturated":                 # too much light/pressure: pinned at the rail
        return 4060 + AMPLITUDE * _pulse_train(n, bpm, rng, 0.03) + noise, 0.0
    if kind == "noise":                     # band-limited noise, no pulse at all
        walk = np.convolve(rng.normal(0, 1, n), np.ones(5) / 5, mode="same")
        return BASELINE + 250 * walk + rng.normal(0, 30, n), 0.0
    if kind == "motion":                    # pulse plus big baseline swings and bursts
        sway = 700 * np.sin(2 * np.pi * 0.25 * t) + 400 * np.sin(2 * np.pi * 0.6 * t + 1)
        bursts = np.zeros(n)
        for start in rng.choice(n - 40, size=seconds // 3, replace=False):
            bursts[start:start + 40] = rng.normal(0, 400, 40)
        pulse = AMPLITUDE * _pulse_train(n, bpm, rng, 0.03)
        return BASELINE + sway + bursts + pulse + noise, bpm
    if kind == "irregular":                 # beats arrive at random intervals
        return BASELINE + AMPLITUDE * _pulse_train(n, bpm, rng, 0.45) + noise, bpm
    raise ValueError(f"unknown kind {kind!r}")


def make_quality_demo_data(seed=1):
    """
    Build the full scripted data set.

    Parameters
    ----------
    seed : int
        Random seed (same seed = same data every run).

    Returns
    -------
    samples : list of int
        Raw 12-bit ADC samples, 50 per second.
    rates : list of float
        MCU pulse rate (BPM) for each sample, in the format sim_packet expects
        (0 where the ESP32 would report no valid pulse).

    Author: Bianca Dibden
    """
    rng = np.random.default_rng(seed)
    samples, rates = [], []
    for _name, kind, seconds, bpm, _expect in SCHEDULE:
        seg, mcu_hr = _segment(kind, seconds, bpm, rng)
        samples.extend(int(v) for v in np.clip(np.round(seg), 0, 4095))
        rates.extend([mcu_hr] * len(seg))
    return samples, rates


if __name__ == "__main__":
    print("Scripted bad-signal timeline (loops after the last row):\n")
    t = 0
    for name, _kind, seconds, _bpm, expect in SCHEDULE:
        print(f"  {t:3d}-{t + seconds:3d} s  {name:26s} {expect}")
        t += seconds
    print(f"\n  total {t} s")
