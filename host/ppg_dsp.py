"""
Host-side signal processing for the Pulse Sensor System (BMET2922 Major Project).

Takes the 50 raw PPG samples that arrive in each 1 s Bluetooth packet and produces:
  - a band-pass filtered waveform (removes baseline wander, Reqt 19),
  - a host-calculated heart rate from adaptive-threshold beat detection,
  - an FFT-based heart rate used as an independent cross-check,
  - a signal-quality verdict (no finger / saturated / implausible HR / methods disagree),
so the GUI can show "signal poor" instead of a nonsense heart rate.

Real-time processing is causal and keeps state between packets, so there are no
edge artefacts at the 1 s packet boundaries. A zero-phase offline function is
also provided for analysing a whole recording (summary report on exit).

Typical use (one call per received packet):

    from ppg_dsp import PPGProcessor
    proc = PPGProcessor()
    result = proc.process_packet(samples)   # samples = list of 50 ints
    result.host_hr       # float BPM, or None if the signal is not trustworthy
                         # (see SHOW_HR_WHEN_SIGNAL_POOR)
    result.filtered      # 50 filtered samples for plotting
    result.quality.ok    # bool; result.quality.reasons explains why not

Call proc.reset() after a reconnection or a run of lost packets.

Author: Bianca Dibden
Date: 15-Oct-2026
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sps

# ---------------------------------------------------------------------------
# System constants
# ---------------------------------------------------------------------------
FS = 50                 # sample rate (Hz), MCU samples every 20 ms (Reqt 1)
SAMPLES_PER_PACKET = 50  # 1 s of data per packet (Reqt 18)
ADC_MAX = 4095          # ESP32 12-bit ADC full scale

# Physiological limits used throughout
HR_MIN_BPM = 30.0
HR_MAX_BPM = 220.0

# ---------------------------------------------------------------------------
# Tunable parameters (check these on the real sensor)
# ---------------------------------------------------------------------------
BANDPASS_LOW_HZ = 0.5    # below this: respiration, motion, contact-pressure drift
BANDPASS_HIGH_HZ = 5.0   # above this: mains pickup / sensor noise; keeps 1st harmonics
BANDPASS_ORDER = 2       # Butterworth prototype order (band-pass is 2x this = 4th order)

THRESH_WINDOW_S = 2.0    # rolling window for adaptive threshold min/max (same as the MCU)
THRESH_HIGH_FRAC = 0.6   # beat registered when signal rises above min + 0.6*range
THRESH_LOW_FRAC = 0.4    # re-armed when signal falls below min + 0.4*range (hysteresis)
                         # (wider than the MCU's 0.55/0.45: 0.55/0.45 gave false beats on
                         #  the drifting-baseline test; the host can afford the wider band)
THRESH_PCTL = 10         # 'min'/'max' are the 10th/90th percentiles (as on the MCU): a brief
                         # motion or pressure spike cannot drag the threshold off the beats
REFRACTORY_S = 60.0 / HR_MAX_BPM   # ignore crossings closer than one beat at 220 BPM
HR_IBI_COUNT = 6         # HR = 60 / median of the last N inter-beat intervals
HR_MIN_IBIS = 3          # need at least this many intervals before reporting an HR
HR_MAX_AGE_S = 4.0       # no beat for this long -> HR is stale

FFT_WINDOW_S = 8.0       # FFT analysis window (8 s -> 0.125 Hz bins before zero-padding)
FFT_NFFT = 2048          # zero-padded length (0.024 Hz / 1.5 BPM interpolation)

FLAT_P2P_COUNTS = 20     # beat detector: window peak-to-peak below this = no pulse, no beats declared
# Testing aid: when True, the beat-detector HR is still reported (host_hr) while the
# signal is judged poor, so the GUI keeps showing HR and the trend. The quality
# verdict (quality.ok / message) is unchanged and still says "Heart rate unreliable".
# Set False to withhold HR whenever the signal is not trustworthy (Reqt 10).
SHOW_HR_WHEN_SIGNAL_POOR = True

SATURATION_MARGIN = 1    # samples within this many counts of 0 or 4095 are clipped
SATURATION_FRAC = 0.20   # more than 20 % clipped samples = saturated
                         # (was 5 counts / 5 %: flagged real signals as saturated;
                         #  a genuinely railed sensor sits at the limit for far longer)
HR_AGREE_BPM = 12.0      # peak HR and FFT HR must agree within this
MIN_SPECTRAL_PURITY = 0.40  # fraction of band power near the dominant peak
                            # (tested: real pulses >= 0.47, band-limited noise ~0.27)
MAX_IBI_CV = 0.20        # max coefficient of variation of recent beat intervals
                         # (tested: real pulses <= 0.08, noise ~0.25)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class SignalQuality:
    """
    Verdict on whether the current signal window can be trusted.

    Attributes
    ----------
    ok : bool
        True if no problems were found.
    reasons : list of str
        Short machine-readable codes, e.g. 'saturated', 'hr_out_of_range',
        'hr_disagree', 'irregular', 'low_spectral_purity', 'no_beats', 'warming_up'.
    spectral_purity : float
        Fraction of 0.5-3.5 Hz power within +/-0.15 Hz of the dominant peak (0-1).
    peak_to_peak : float
        Peak-to-peak amplitude of the filtered window (ADC counts).
    clipped_frac : float
        Fraction of raw samples at the ADC rails (0-1).
    hr_peak, hr_fft : float or None
        The two heart-rate estimates compared by the quality check (BPM).
    ibi_cv : float or None
        Coefficient of variation of the recent beat intervals.

    Author: Bianca Dibden
    """
    ok: bool = False
    reasons: list = field(default_factory=list)
    spectral_purity: float = 0.0
    peak_to_peak: float = 0.0
    clipped_frac: float = 0.0
    hr_peak: float = None
    hr_fft: float = None
    ibi_cv: float = None

    def threshold_lines(self):
        """
        Measured value vs. threshold for each FAILED check only (Researcher tab).

        Returns
        -------
        list of str
            One line per failed reason; empty when the signal is OK.

        Author: Bianca Dibden
        """
        fmt = lambda v: "--" if v is None else f"{v:.1f}"
        lines = []
        for r in self.reasons:
            if r == "saturated":
                lines.append(f"Clipped samples: {self.clipped_frac:.0%} (limit {SATURATION_FRAC:.0%})")
            elif r == "hr_out_of_range":
                lines.append(f"Peak HR: {fmt(self.hr_peak)} BPM (allowed {HR_MIN_BPM}-{HR_MAX_BPM})")
            elif r == "hr_disagree":
                lines.append(f"Peak HR {fmt(self.hr_peak)} vs FFT HR {fmt(self.hr_fft)} BPM "
                             f"(max difference {HR_AGREE_BPM:.0f})")
            elif r == "irregular":
                lines.append(f"Beat interval variation: {fmt(self.ibi_cv)} (limit {MAX_IBI_CV})")
            elif r == "low_spectral_purity":
                lines.append(f"Spectral purity: {self.spectral_purity:.2f} (min {MIN_SPECTRAL_PURITY})")
        return lines

    def message(self):
        """
        Human-readable summary for the GUI / system log.

        Returns
        -------
        str
            e.g. 'Signal OK' or 'Heart rate unreliable: saturated (too much pressure/light)'.

        Author: Bianca Dibden
        """
        if self.ok:
            return "Signal OK"
        text = {
            "saturated": "saturated (too much pressure/light)",
            "hr_out_of_range": "pulse rate out of range",
            "hr_disagree": "pulse methods disagree (motion?)",
            "low_spectral_purity": "no clear pulse rhythm",
            "irregular": "irregular beat intervals (motion/noise?)",
            "warming_up": "acquiring signal",
            "no_beats": "no beats detected",
        }
        return "Heart rate unreliable: " + ", ".join(text.get(r, r) for r in self.reasons)


@dataclass
class ProcessedPacket:
    """
    Everything the GUI needs from one processed packet.

    Attributes
    ----------
    raw : np.ndarray
        The 50 raw samples as received (ADC counts).
    filtered : np.ndarray
        The 50 band-pass filtered samples (ADC counts, zero-centred).
    threshold : np.ndarray
        Adaptive upper threshold for each filtered sample (for plotting).
    beat_times : list of float
        Times (s since processor start/reset) of beats detected in this packet.
    hr_peak : float or None
        HR from adaptive-threshold beat detection (BPM).
    hr_fft : float or None
        HR from the dominant spectral peak over the last FFT_WINDOW_S seconds (BPM).
    host_hr : float or None
        The HR to display/alarm on. Equal to hr_peak when the signal is valid,
        otherwise None.
    quality : SignalQuality
        Signal validation result.

    Author: Bianca Dibden
    """
    raw: np.ndarray
    filtered: np.ndarray
    threshold: np.ndarray
    beat_times: list
    hr_peak: float | None
    hr_fft: float | None
    host_hr: float | None
    quality: SignalQuality


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class StreamingBandpass:
    """
    Causal Butterworth band-pass filter that keeps its internal state between
    calls, so a signal fed in 1 s chunks is filtered exactly as if it were one
    continuous stream (no discontinuities at packet boundaries).

    Implemented as second-order sections (SOS) for numerical stability.

    Author: Bianca Dibden
    """

    def __init__(self, fs=FS, low=BANDPASS_LOW_HZ, high=BANDPASS_HIGH_HZ,
                 order=BANDPASS_ORDER):
        """
        Design the filter.

        Parameters
        ----------
        fs : float
            Sample rate (Hz).
        low, high : float
            -3 dB cut-off frequencies (Hz).
        order : int
            Butterworth prototype order (the band-pass has 2*order poles).

        Author: Bianca Dibden
        """
        self.sos = sps.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
        self._zi_unit = sps.sosfilt_zi(self.sos)
        self.zi = None

    def reset(self):
        """
        Forget the filter state (call after a gap in the data).

        Author: Bianca Dibden
        """
        self.zi = None

    def process(self, chunk):
        """
        Filter one chunk of samples, continuing from the previous state.

        Parameters
        ----------
        chunk : array-like
            New raw samples.

        Returns
        -------
        np.ndarray
            Filtered samples, same length as chunk.

        Author: Bianca Dibden
        """
        x = np.asarray(chunk, dtype=float)
        if self.zi is None:
            # Start in steady state for the first sample's DC level, which avoids
            # a large start-up transient from the ~2000-count sensor offset.
            self.zi = self._zi_unit * x[0]
        y, self.zi = sps.sosfilt(self.sos, x, zi=self.zi)
        return y


class AdaptiveBeatDetector:
    """
    Beat detector with an adaptive, hysteretic threshold.

    For every filtered sample the threshold is recomputed from the spread of
    the last THRESH_WINDOW_S seconds:
        lo = 10th percentile, hi = 90th percentile of the window
        thr_high = lo + THRESH_HIGH_FRAC * (hi - lo)
        thr_low  = lo + THRESH_LOW_FRAC  * (hi - lo)
    Percentiles are used instead of the true min/max so that a short artefact
    (e.g. the filter's response to a sudden change in finger pressure) does not
    raise the threshold above the real beats for the next few seconds.
    A beat is the rising crossing of thr_high (the same "rising edge of the pulse
    train" idea as Labs 3-4). The detector only re-arms once the signal drops
    below thr_low (hysteresis, the software equivalent of switch debouncing), and
    crossings within REFRACTORY_S of the previous beat are ignored.

    Author: Bianca Dibden
    """

    def __init__(self, fs=FS):
        """
        Parameters
        ----------
        fs : float
            Sample rate (Hz).

        Author: Bianca Dibden
        """
        self.fs = fs
        self.window = deque(maxlen=int(THRESH_WINDOW_S * fs))
        self.reset()

    def reset(self):
        """
        Clear all detector state.

        Author: Bianca Dibden
        """
        self.window.clear()
        self.sample_index = 0
        self.armed = False
        self.last_beat_index = None
        self.ibis = deque(maxlen=HR_IBI_COUNT)   # inter-beat intervals (s)
        self.beat_times_all = []                 # every beat time since reset (s)

    def process(self, filtered, min_p2p=FLAT_P2P_COUNTS):
        """
        Run the detector over a chunk of filtered samples.

        Parameters
        ----------
        filtered : array-like
            Band-pass filtered samples.
        min_p2p : float
            Window peak-to-peak below which no beats are declared (flat signal).

        Returns
        -------
        thresholds : np.ndarray
            thr_high for each sample (for plotting alongside the waveform).
        new_beats : list of float
            Beat times (s since reset) found in this chunk.

        Author: Bianca Dibden
        """
        thresholds = np.empty(len(filtered))
        new_beats = []
        refractory = int(REFRACTORY_S * self.fs)

        for i, x in enumerate(filtered):
            # a) Slide the window forward by one sample and recompute the two
            #    threshold levels from its low/high percentiles.
            self.window.append(x)
            lo, hi = np.percentile(self.window, [THRESH_PCTL, 100 - THRESH_PCTL])
            rng = hi - lo
            thr_high = lo + THRESH_HIGH_FRAC * rng
            thr_low = lo + THRESH_LOW_FRAC * rng
            thresholds[i] = thr_high

            # b) Only look for beats if the window has a real pulse-sized swing.
            if rng >= min_p2p:
                # c) Arm when the signal is low; fire when it then rises above
                #    the upper level (same armed/upper/lower idea as the MCU).
                if x < thr_low:
                    self.armed = True
                elif self.armed and x > thr_high:
                    self.armed = False
                    n = self.sample_index
                    # d) Refractory check, then store the interval between this
                    #    beat and the previous one.
                    if self.last_beat_index is None or n - self.last_beat_index >= refractory:
                        if self.last_beat_index is not None:
                            ibi = (n - self.last_beat_index) / self.fs
                            if 60.0 / HR_MAX_BPM <= ibi <= 60.0 / HR_MIN_BPM:
                                self.ibis.append(ibi)
                        self.last_beat_index = n
                        t = n / self.fs
                        new_beats.append(t)
                        self.beat_times_all.append(t)
            self.sample_index += 1

        return thresholds, new_beats

    def heart_rate(self):
        """
        Current HR from the median of the most recent inter-beat intervals.

        The median (rather than the mean) means one missed or extra beat does
        not drag the reading. Returns None if there are fewer than HR_MIN_IBIS
        valid intervals or if the last beat is older than HR_MAX_AGE_S.

        Returns
        -------
        float or None
            Heart rate in BPM.

        Author: Bianca Dibden
        """
        if len(self.ibis) < HR_MIN_IBIS or self.last_beat_index is None:
            return None
        if (self.sample_index - self.last_beat_index) / self.fs > HR_MAX_AGE_S:
            return None
        return 60.0 / float(np.median(self.ibis))


def fft_heart_rate(x, fs=FS, nfft=FFT_NFFT, fmin=HR_MIN_BPM / 60.0, fmax=3.5):
    """
    Estimate heart rate from the dominant frequency of a signal window.

    The window is mean-removed, multiplied by a Hann window (reduces spectral
    leakage), zero-padded to nfft points (interpolates the spectrum), and the
    largest peak of |X(f)|^2 between fmin and fmax is taken as the pulse
    frequency: HR = 60 * f_peak.

    Parameters
    ----------
    x : array-like
        Signal window (ideally already band-pass filtered).
    fs : float
        Sample rate (Hz).
    nfft : int
        FFT length after zero-padding.
    fmin, fmax : float
        Search band (Hz). Default 0.5-3.5 Hz = 30-210 BPM.

    Returns
    -------
    hr : float or None
        Heart rate (BPM), None if the window is too short.
    purity : float
        Fraction of in-band power within +/-0.15 Hz of the peak (0-1). A clean
        periodic pulse gives a high value; noise or motion gives a low one.

    Author: Bianca Dibden
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 2 * fs:
        return None, 0.0
    x = x - x.mean()
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=max(nfft, len(x)))) ** 2
    freqs = np.fft.rfftfreq(max(nfft, len(x)), d=1.0 / fs)
    band = (freqs >= fmin) & (freqs <= fmax)
    if not band.any() or spec[band].sum() == 0:
        return None, 0.0
    f_band, p_band = freqs[band], spec[band]
    f_peak = f_band[np.argmax(p_band)]
    near = np.abs(f_band - f_peak) <= 0.15
    purity = float(p_band[near].sum() / p_band.sum())
    return 60.0 * float(f_peak), purity


def assess_signal(raw, filtered, hr_peak, hr_fft, purity, ibis=None):
    """
    Decide whether a window of signal is trustworthy.

    Checks, in order:
      saturated   - > SATURATION_FRAC of raw samples at the ADC rails
      no_beats    - beat detector has no current HR
      hr_out_of_range - HR outside HR_MIN_BPM..HR_MAX_BPM
      hr_disagree - |hr_peak - hr_fft| > HR_AGREE_BPM (also accepts the FFT
                    locking onto the 2nd harmonic, i.e. hr_fft ~ 2*hr_peak)
      irregular   - coefficient of variation (std/mean) of the recent beat
                    intervals > MAX_IBI_CV. Noise produces random intervals;
                    a real pulse is nearly periodic over a few beats.
      low_spectral_purity - purity < MIN_SPECTRAL_PURITY

    Parameters
    ----------
    raw : array-like
        Recent raw samples (ADC counts).
    filtered : array-like
        The same samples after band-pass filtering.
    hr_peak, hr_fft : float or None
        Heart-rate estimates from the two methods (BPM).
    purity : float
        Spectral purity from fft_heart_rate().
    ibis : array-like, optional
        Recent inter-beat intervals (s) for the regularity check.

    Returns
    -------
    SignalQuality

    Author: Bianca Dibden
    """
    raw = np.asarray(raw, dtype=float)
    filtered = np.asarray(filtered, dtype=float)
    p2p = float(np.ptp(filtered[-FS * 3:])) if len(filtered) else 0.0
    reasons = []
    hr_ibi_cv = None

    clipped = np.mean((raw <= SATURATION_MARGIN) | (raw >= ADC_MAX - SATURATION_MARGIN))
    if clipped > SATURATION_FRAC:
        reasons.append("saturated")
    if not reasons:
        if hr_peak is None:
            reasons.append("no_beats")
        elif not HR_MIN_BPM <= hr_peak <= HR_MAX_BPM:
            reasons.append("hr_out_of_range")
        elif hr_fft is not None:
            agree = (abs(hr_peak - hr_fft) <= HR_AGREE_BPM or
                     abs(2 * hr_peak - hr_fft) <= HR_AGREE_BPM)
            if not agree:
                reasons.append("hr_disagree")
        if ibis is not None and len(ibis) >= 3:
            ibis = np.asarray(ibis, dtype=float)
            hr_ibi_cv = float(ibis.std() / ibis.mean())
            if hr_ibi_cv > MAX_IBI_CV:
                reasons.append("irregular")
        if hr_fft is not None and hr_peak is not None:
            if purity < MIN_SPECTRAL_PURITY:
                reasons.append("low_spectral_purity")

    return SignalQuality(ok=not reasons, reasons=reasons,
                         spectral_purity=purity, peak_to_peak=p2p,
                         clipped_frac=float(clipped), hr_peak=hr_peak, hr_fft=hr_fft,
                         ibi_cv=hr_ibi_cv)


# ---------------------------------------------------------------------------
# Real-time processor (one call per packet)
#
# DATA FLOW for each 1 s packet (50 raw samples from comms.py):
#
#   raw samples
#     |
#     +--> StreamingBandpass.process()      0.5-5 Hz band-pass, keeps state between packets
#     |        -> filtered samples (zero-centred; used for plotting AND everything below)
#     |
#     +--> AdaptiveBeatDetector.process()   per sample: threshold from the last 2 s (P10/P90),
#     |        |                            rising crossing = beat, hysteresis + refractory
#     |        +--> inter-beat intervals (IBIs)
#     |        +--> heart_rate(): 60 / median(last 6 IBIs)      => hr_peak
#     |
#     +--> fft_heart_rate()                 last 8 s of filtered data, dominant peak => hr_fft
#     |                                     (+ "spectral purity" = how clean the rhythm is)
#     |
#     +--> assess_signal()                  saturated? hr_peak vs hr_fft agree?
#                                           IBIs regular? purity high?  => SignalQuality
#
#   host_hr = hr_peak (shown on the GUI; the FFT value is only a cross-check)
#
# The MCU (A2_Lumi.ino) does the first two stages in a simpler form; it has no
# FFT and no quality checks. See the Research note "ESP32 vs Host Signal Processing".
# ---------------------------------------------------------------------------
class PPGProcessor:
    """
    Real-time PPG pipeline for the host: packet samples -> band-pass filter ->
    adaptive-threshold beats -> HR, with FFT cross-check and signal validation.

    Author: Bianca Dibden
    """

    def __init__(self, fs=FS):
        """
        Parameters
        ----------
        fs : float
            Sample rate (Hz).

        Author: Bianca Dibden
        """
        self.fs = fs
        self.bandpass = StreamingBandpass(fs)
        self.detector = AdaptiveBeatDetector(fs)
        n_hist = int(FFT_WINDOW_S * fs)
        self.raw_hist = deque(maxlen=n_hist)
        self.filt_hist = deque(maxlen=n_hist)
        self.samples_seen = 0

    def reset(self):
        """
        Restart the pipeline from scratch. Call after a reconnection or after
        lost packets, so a gap in the data is not treated as continuous signal.

        Author: Bianca Dibden
        """
        self.bandpass.reset()
        self.detector.reset()
        self.raw_hist.clear()
        self.filt_hist.clear()
        self.samples_seen = 0

    def process_packet(self, samples):
        """
        Process one packet's worth of raw samples.

        Parameters
        ----------
        samples : array-like
            Raw ADC samples (normally 50, oldest first).

        Returns
        -------
        ProcessedPacket

        Author: Bianca Dibden
        """
        raw = np.asarray(samples, dtype=float)
        # 1. Remove baseline drift and high-frequency noise (state carries over,
        #    so packet boundaries are invisible to the filter).
        filtered = self.bandpass.process(raw)
        # 2. Find beats in the filtered signal, sample by sample.
        thresholds, beats = self.detector.process(filtered)

        # 3. Keep the last 8 s of data: the FFT and the quality checks need
        #    more than the 1 s that arrives in each packet.
        self.raw_hist.extend(raw)
        self.filt_hist.extend(filtered)
        self.samples_seen += len(raw)

        # 4. Two independent HR estimates: beat intervals (primary) and FFT peak
        #    (cross-check).
        hr_peak = self.detector.heart_rate()
        hr_fft, purity = fft_heart_rate(np.array(self.filt_hist), self.fs)

        # 5. Decide whether to trust them (the first 3 s are always "warming up").
        if self.samples_seen < 3 * self.fs:
            # Filter and threshold need a few seconds to settle.
            quality = SignalQuality(ok=False, reasons=["warming_up"])
        else:
            quality = assess_signal(self.raw_hist, self.filt_hist, hr_peak, hr_fft,
                                    purity, ibis=list(self.detector.ibis))

        return ProcessedPacket(
            raw=raw, filtered=filtered, threshold=thresholds, beat_times=beats,
            hr_peak=hr_peak, hr_fft=hr_fft,
            host_hr=hr_peak if (quality.ok or SHOW_HR_WHEN_SIGNAL_POOR) else None,
            quality=quality,
        )


# ---------------------------------------------------------------------------
# Offline analysis (whole recording, e.g. for the summary PDF on exit)
# ---------------------------------------------------------------------------
def analyse_recording(raw, fs=FS):
    """
    Analyse a complete recording offline.

    Uses zero-phase filtering (filtfilt: forward then backward), which has no
    delay and no phase distortion but needs the whole record, so it is only
    suitable after recording has finished.

    Parameters
    ----------
    raw : array-like
        Complete raw recording (ADC counts).
    fs : float
        Sample rate (Hz).

    Returns
    -------
    dict
        'filtered'     : zero-phase filtered signal
        'beat_times'   : beat times (s)
        'ibi'          : inter-beat intervals (s)
        'hr_series'    : instantaneous HR for each interval (BPM)
        'mean_hr'      : mean HR (BPM)
        'sdnn_ms'      : standard deviation of intervals (HRV, ms)
        'rmssd_ms'     : RMS of successive interval differences (HRV, ms)
        'hr_fft'       : HR from the spectrum of the whole record (BPM)

    Author: Bianca Dibden
    """
    raw = np.asarray(raw, dtype=float)
    sos = sps.butter(BANDPASS_ORDER, [BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ],
                     btype="bandpass", fs=fs, output="sos")
    filtered = sps.sosfiltfilt(sos, raw)

    # Systolic peaks: at least one 220-BPM period apart, and prominent relative
    # to the signal's spread.
    peaks, _ = sps.find_peaks(filtered, distance=int(REFRACTORY_S * fs),
                              prominence=0.5 * np.std(filtered))
    beat_times = peaks / fs
    ibi = np.diff(beat_times)
    ibi = ibi[(ibi >= 60.0 / HR_MAX_BPM) & (ibi <= 60.0 / HR_MIN_BPM)]

    out = {
        "filtered": filtered,
        "beat_times": beat_times,
        "ibi": ibi,
        "hr_series": 60.0 / ibi if len(ibi) else np.array([]),
        "mean_hr": 60.0 / ibi.mean() if len(ibi) else None,
        "sdnn_ms": 1000 * ibi.std(ddof=1) if len(ibi) > 1 else None,
        "rmssd_ms": 1000 * np.sqrt(np.mean(np.diff(ibi) ** 2)) if len(ibi) > 2 else None,
        "hr_fft": fft_heart_rate(filtered, fs, nfft=max(FFT_NFFT, len(filtered)))[0],
    }
    return out
