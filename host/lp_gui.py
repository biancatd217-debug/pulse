"""LumiPulse PPG Monitor - BMET2922 Major Project.

Monitor shows Host HR, PPG, alarms, recording state and the system log.
Researcher compares MCU/Host HR and displays signal analysis and processing details.

PPGLink receives and validates packets, then runs ppg_dsp.py before queueing records.
The GUI displays these results; Researcher analysis does not replace Host HR detection.
Recording follows the MCU packet flag and produces a CSV and a PDF session summary.

Run from the project root:
    python3 host/lp_gui.py SIM
    python3 host/lp_gui.py COM5
    python3 host/lp_gui.py /dev/tty.BMET2922-G00

Use the actual Bluetooth serial port for hardware; no argument defaults to SIM.
Integration with comms.py / ppg_dsp.py: Bianca Dibden
"""

# ============================================================
# IMPORTS
# ============================================================

import FreeSimpleGUI as sg
import numpy as np
import time
import csv
import os
import sys
import textwrap
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from comms import PPGLink, STATUS_CONNECTED, STATUS_CONNECTING, MAX_FORWARD_GAP
from packet_format import SAMPLES_PER_PACKET, SEQ_MODULO

# ============================================================
# HOW THIS FILE IS ORGANISED (read this first)
# ============================================================
# DATA FLOW
#   MCU / simulator --> comms.py PPGLink (background thread: validates packets,
#   runs ppg_dsp.py filtering + Host HR) --> queue of HostRecord objects
#   --> main() event loop below (polls the queue every 50 ms) --> plots,
#   labels, CSV file and PDF report.
#
# A HostRecord (one packet, normally 50 samples = 1 s) holds:
#   seq, time, raw[], filtered[], mcu_hr, host_hr, signal_ok, signal_message,
#   recording (button flag from the MCU), latency_s.
#
# KEY COMPONENTS (top to bottom)
#   1. Constants            colours, 50 Hz sample rate, display windows, HR limits
#   2. add_plot()           builds one themed Matplotlib figure inside a Tk canvas
#   3. Researcher analysis  FFT, spectrogram, 2nd derivative (display only)
#   4. Gap detection        display_packet_gap() decides if data was lost
#   5. SessionSummary +     collects stats while recording, then a background
#      generate_session_pdf thread writes the PDF report from the saved CSV
#   6. Log / comms display  system log (Reqt 13), Bluetooth status (Reqt 14)
#   7. CSV recording        start_recording / save_packet / stop_recording
#   8. HR status            colours + text for low / normal / high HR
#   9. Layout helpers       configure_*() set up resizing after the window exists
#  10. create_gui()         defines every widget (Monitor tab + Researcher tab)
#  11. Display helpers      legend styling, y-axis autoscale, update_hr_display()
#  12. main()               creates plots + PPGLink, then runs the event loop
#
# NAMING CONVENTION: every widget has a key like "-CURRENT_HR-"; the code reads
# or changes it with window["-CURRENT_HR-"].update(...).

# ============================================================
# GUI COLOURS AND FONTS
# ============================================================

BG = "#20252B"
PANEL = "#292F36"
BORDER = "#3B434C"
TEXT = "#E8EAED"
TEXT_MUTED = "#AEB6BF"
ACCENT = "#6FA8DC"
GREEN = "#7BC47F"
AMBER = "#E6B566"
RED = "#E57373"

FONT_TITLE = ("Helvetica", 24, "bold")
FONT_SECTION = ("Helvetica", 13, "bold")
FONT_NORMAL = ("Helvetica", 12)
FONT_SMALL = ("Helvetica", 12, "bold")
FONT_HR = ("Helvetica", 80, "bold")
FONT_VALUE = ("Helvetica", 16, "bold")

# ============================================================
# SYSTEM SETTINGS
# ============================================================

# Nominal MCU sampling is 50 Hz; host display calculations assume 20 ms spacing.
SAMPLE_RATE = 50
SAMPLE_PERIOD = 1 / SAMPLE_RATE

# PPG shows the most recent 10 seconds.
PPG_DISPLAY_SECONDS = 10
PPG_BUFFER_SIZE = SAMPLE_RATE * PPG_DISPLAY_SECONDS


# HR trend shows most recent 60 seconds.
HR_TREND_SECONDS = 60

# Default HR limits. User can override these.
LOW_HR_LIMIT = 50
HIGH_HR_LIMIT = 100

# Communication alarm (5 s, Reqt 14) is timed in comms.py (ALARM_AFTER_S).

# Port used when none is given on the command line ("SIM" = simulator).
DEFAULT_PORT = "SIM"

# Time to wait for GUI events (ms); this does not set the plot update rate.
GUI_POLL_MS = 50

# ============================================================
# MATPLOTLIB HELPER
# ============================================================

def add_plot(canvas_element, figsize):
    """Create a themed Matplotlib figure and embed it in a GUI canvas.

    FreeSimpleGUI has no native plotting, so each graph is a sg.Canvas
    placeholder; this function puts a Matplotlib figure inside it.
    Returns (figure, axes, canvas_agg). Later code updates the line data and
    calls canvas_agg.draw() to refresh the picture.
    """

    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor(PANEL)
    ax.set_facecolor(PANEL)
    ax.tick_params(
        colors=TEXT_MUTED,
        labelsize=10
    )

    ax.xaxis.label.set_color(TEXT_MUTED)
    ax.yaxis.label.set_color(TEXT_MUTED)

    for spine in ax.spines.values():
        spine.set_color(BORDER)

    # Keep the embedded plot from changing the space allocated by the GUI layout.
    canvas_element.TKCanvas.pack_propagate(False)

    fig_agg = FigureCanvasTkAgg(
        fig,
        canvas_element.TKCanvas
    )

    fig_agg.draw()
    # Let TkAgg handle resizing and combine pending redraw requests.
    fig_agg.get_tk_widget().pack(
        side="top",
        fill="both",
        expand=True
    )

    return fig, ax, fig_agg

# ============================================================
# RESEARCHER ANALYSIS (visualisation only; no HR/filter replacement)
# ============================================================

def spectral_magnitude(samples):
    """Return frequencies (Hz) and one-sided FFT amplitudes (filtered ADC counts)."""
    x = np.asarray(samples, dtype=float)
    if len(x) < 2 * SAMPLE_RATE or not np.isfinite(x).all():
        return np.array([]), np.array([])
    # Remove the average; the Hann window reduces spreading into nearby frequencies.
    # Divide by the window sum to account for the Hann window when scaling amplitude.
    window = np.hanning(len(x))
    magnitude = 2 * np.abs(np.fft.rfft((x - x.mean()) * window)) / window.sum()
    # Undo doubling at 0 Hz and (for even sample counts) 25 Hz: these have no matching negative frequency.
    magnitude[0] *= .5
    if len(x) % 2 == 0:
        magnitude[-1] *= .5
    frequency = np.fft.rfftfreq(len(x), SAMPLE_PERIOD)
    keep = frequency <= 5
    return frequency[keep], magnitude[keep]


def ppg_spectrogram(samples):
    """Analyse filtered PPG with 4 s Hann windows and a 0.5 s hop (87.5% overlap).

    Return frequencies (Hz), window-centre times relative to the newest sample,
    and amplitude in dB relative to 1 ADC count. Per-window means are removed.
    Zeros extend the history edges; edge estimates therefore include assumed data.
    """
    x = np.asarray(samples, dtype=float)
    length, hop = 200, 25
    if len(x) < length or not np.isfinite(x).all():
        return None
    # Boundary windows include assumed zeros, not extra measured PPG samples.
    padded = np.pad(x, (length // 2, length // 2))  # Add 2 s of zeros at each end.
    frames = np.lib.stride_tricks.sliding_window_view(padded, length)[::hop]  # 4 s windows, 0.5 s apart.
    window = np.hanning(length)  # Smooth each window's edges to reduce spectral leakage.
    # Remove each window's average, apply the Hann window, then calculate FFT amplitudes.
    magnitude = 2 * np.abs(np.fft.rfft(
        (frames - frames.mean(axis=1, keepdims=True)) * window, axis=1
    )) / window.sum()
    magnitude[:, 0] *= .5
    magnitude[:, -1] *= .5
    frequency = np.fft.rfftfreq(length, SAMPLE_PERIOD)
    keep = frequency <= 5
    # Locate each window's centre relative to the newest sample (0 s); older times are negative.
    centres = (np.arange(len(frames)) * hop + (length - 1) / 2 - length // 2 - (len(x) - 1)) * SAMPLE_PERIOD
    # Arrange rows by frequency and columns by time; floor amplitudes to avoid log(0).
    return frequency[keep], centres, 20 * np.log10(np.maximum(magnitude[:, keep].T, 1e-6))


def display_packet_gap(previous, record, interrupted=False, timeout=5.0):
    """Return 0 for continuity, missing sample slots for a known gap, or None if uncertain.

    Use sequence numbers and host time to detect interruptions, not exact acquisition times.
    These results affect display history only; no replacement measurements are created.
    """
    if interrupted or len(record.raw) != SAMPLES_PER_PACKET or len(record.filtered) != SAMPLES_PER_PACKET:
        return None
    if previous is None:
        return 0
    if len(previous.raw) != SAMPLES_PER_PACKET or len(previous.filtered) != SAMPLES_PER_PACKET:
        return None
    # Treat sequence 65535 -> 0 as normal wraparound, not a missing packet.
    step = (record.seq - previous.seq) % SEQ_MODULO
    if 1 < step <= MAX_FORWARD_GAP:
        return (step - 1) * SAMPLES_PER_PACKET
    if step != 1 or record.time - previous.time >= timeout:
        return None
    return 0


class ResearcherAnalysis:
    """Keep up to 10 s of filtered samples, clearing history after detected interruptions."""
    def __init__(self):
        # deque(maxlen) automatically discards the oldest sample when full (500 samples = 10 s).
        self.samples = deque(maxlen=10 * SAMPLE_RATE)
        self.last_seq = None
        self.valid_samples = 0

    def append(self, record, interrupted=False):
        # Called once per packet. A sequence jump or reconnect restarts the history so
        # the FFT/spectrogram never join samples from either side of a gap.
        if interrupted or (self.last_seq is not None and (record.seq - self.last_seq) % 65536 != 1):
            self.samples.clear()
            self.valid_samples = 0
        self.last_seq = record.seq
        # Keep actual filtered samples even when quality is poor or HR is unavailable.
        self.samples.extend(record.filtered)
        self.valid_samples = self.valid_samples + len(record.filtered) if record.signal_ok else 0

    def amplitude(self):
        # Recent observed amplitude uses continuous filtered data, regardless
        # of pulse validity or signal quality; reject incomplete/non-finite data.
        samples = np.asarray(self.samples)[-3 * SAMPLE_RATE:]
        if len(samples) < 3 * SAMPLE_RATE or not np.isfinite(samples).all():
            return None
        return float(np.ptp(samples))


def researcher_draw_idle(canvas):
    """Schedule a visible plot to redraw when Tk is idle; hidden plots retain their data."""
    if canvas.get_tk_widget().winfo_ismapped():
        canvas.draw_idle()


def bind_researcher_mapping(canvas):
    """Let TkAgg resize plots, skipping repeated notifications with the same size and DPI.

    Repeated notifications can rebuild the canvas image unnecessarily and cause flashing.
    """
    widget = canvas.get_tk_widget()
    last_size = None

    def resize_if_changed(event):
        nonlocal last_size
        size = (event.width, event.height, canvas.device_pixel_ratio)
        if size == last_size:  # Ignore duplicate resize events to avoid unnecessary canvas rebuilds.
            return
        last_size = size
        canvas.resize(event)

    # Check for a real size change before calling TkAgg's existing resize method.
    widget.bind('<Configure>', resize_if_changed)
    widget.bind('<Map>', lambda event: canvas.draw_idle(), add='+')


def create_researcher_plots(window):
    """Create the Researcher figures and plot objects once, ready for live updates.

    Plots are built once and only their data is replaced afterwards
    (set_data), which is much faster than redrawing from scratch each second.
    """
    result = {}
    specs = (
        ('fft', '-FFT_CANVAS-', 'Frequency Spectrum (Last 10 s)', 'Frequency (Hz)', 'Spectral Amplitude (ADC counts)'),
        ('spectrogram', '-SPECTROGRAM_CANVAS-', 'Spectrogram — Last 10 s', 'Time (s)', 'Frequency (Hz)'),
        ('derivative', '-DERIVATIVE_CANVAS-', 'Second Derivative (PPG Acceleration)', 'Time (s)', 'ADC counts/s^2'),
    )
    for name, key, title, xlabel, ylabel in specs:
        fig, ax, canvas = add_plot(window[key], (4.5, 1.7))
        ax.set_title(title, color=TEXT, fontsize=12)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        if name == 'derivative':
            # Move only the label slightly left; retain axes/figure bounds.
            ax.yaxis.labelpad = 0
        fig.subplots_adjust(left=.17, right=.97, bottom=.24, top=.85)
        if name == 'spectrogram':
            # Fixed cells for the boundary-padded 200-sample / 25-hop STFT.
            # Full-history centres are -9.99..0.01 s; half-hop cell edges
            # extend beyond the visible -10..0 s axis without stretching.
            # Right-align partial histories; unacquired history stays NaN.
            ax.set_position([.16, .24, .60, .61])
            artist = ax.imshow(np.full((21, 21), np.nan), origin='lower', aspect='auto',
                               extent=(-10.24, .26, -.125, 5.125),
                               vmin=-40, vmax=60, cmap='cividis', interpolation='nearest')
            artist.set_visible(False)
            # Explicit colourbar axes avoid automatic parent-axes repositioning.
            colourbar_axes = fig.add_axes([.80, .24, .025, .61])
            bar = fig.colorbar(artist, cax=colourbar_axes)
            bar.set_label('Magnitude (dB re 1 ADC count)', color=TEXT_MUTED, fontsize=8)
            bar.ax.tick_params(colors=TEXT_MUTED, labelsize=8)
            ax.set_xlim(-10, 0)
            ax.set_ylim(0, 5)
        else:
            artist, = ax.plot([], [], color=ACCENT, linewidth=1)
            ax.set_xlim((0, 5) if name == 'fft' else (-10, 0))
            ax.grid(True, alpha=.2)
        waiting = ax.text(.5, .5, 'Acquiring continuous data', transform=ax.transAxes,
                          ha='center', color=TEXT_MUTED, fontsize=10)
        bind_researcher_mapping(canvas)
        # Save the axes, plotted object, canvas and waiting message for later updates.
        result[name] = (ax, artist, canvas, waiting)
    return result


def update_researcher_plots(plots, analysis):
    """Update existing FFT, spectrogram and derivative plots from the current history."""
    x = np.asarray(analysis.samples, dtype=float)
    # FFT: describe the available continuous filtered history, up to ten seconds.
    ax, line, canvas, waiting = plots['fft']
    freq, magnitude = spectral_magnitude(x)
    line.set_data(freq, magnitude)
    waiting.set_visible(len(freq) == 0)
    if len(freq):
        ax.set_ylim(0, max(1, float(magnitude.max()) * 1.15))
    researcher_draw_idle(canvas)

    # Spectrogram: update fixed cells; NaNs mark history not yet acquired.
    ax, image, canvas, waiting = plots['spectrogram']
    spec = ppg_spectrogram(x)
    image.set_visible(spec is not None)
    waiting.set_visible(spec is None)
    if spec is not None:
        freq, centres, db = spec
        display = np.full((21, 21), np.nan)
        display[:, -db.shape[1]:] = db
        image.set_data(display)
    researcher_draw_idle(canvas)

    ax, line, canvas, waiting = plots['derivative']
    if len(x) >= SAMPLE_RATE:
        # Differentiate twice using 0.02 s sample spacing (ADC counts/s^2).
        # Hide two samples at each end, where the derivative is less reliable.
        acceleration = np.gradient(np.gradient(x, SAMPLE_PERIOD, edge_order=2),
                                   SAMPLE_PERIOD, edge_order=2)
        acceleration[:2] = acceleration[-2:] = np.nan
        # Place newest data at 0 s; older samples sit to the left on the fixed axis.
        line.set_data((np.arange(len(x)) - len(x) + 1) * SAMPLE_PERIOD,
                      acceleration)
        autoscale_y(ax, acceleration)
    else:
        line.set_data([], [])
    waiting.set_visible(len(x) < SAMPLE_RATE)
    researcher_draw_idle(canvas)

# ============================================================
# COMPLETED-SESSION PDF REPORTS (no changes to the CSV schema)
# ============================================================

class SessionSummary:
    """Collect recording metadata: saved packets, signal quality, HR limits and observed alarms.

    One instance exists per recording. main() feeds it every saved packet
    (observe), alarm transition (alarm) and limit change (change_limits); the
    PDF generator then prints these totals on the report.
    """
    def __init__(self, filename, low, high):
        self.filename = filename
        self.samples = 0
        self.quality = Counter()
        self.limits = [(0.0, low, high)]
        self.alarm_events = Counter()
        self.last_alarm = None
        self.start_time = None
        self.packet_count = 0
        self.good_packets = 0
        self.valid_samples = 0
        self.gaps = 0
        self.last_seq = None

    def observe(self, record):
        if self.start_time is None:
            self.start_time = record.time - (len(record.raw) - 1) * SAMPLE_PERIOD
        # Observe each saved packet once; quality counters describe packets, not individual beats.
        self.samples += len(record.raw)
        self.packet_count += 1
        self.good_packets += int(record.signal_ok)
        self.valid_samples = self.valid_samples + len(record.filtered) if record.signal_ok else 0
        # Count each failed check on its own so combined faults don't split the tally.
        for reason in record.signal_reasons or (record.signal_message,):
            self.quality[reason] += 1
        if self.last_seq is not None and (record.seq - self.last_seq) % 65536 != 1:
            # Count sequence discontinuities, not the number of missing packets.
            self.gaps += 1
            self.valid_samples = len(record.filtered) if record.signal_ok else 0
        self.last_seq = record.seq

    def alarm(self, state):
        # Count entries into observed low/high states, not every out-of-limit packet.
        if state in ('low', 'high') and state != self.last_alarm:
            self.alarm_events[state] += 1
        self.last_alarm = state

    def change_limits(self, low, high):
        # Timestamp limit changes relative to the reconstructed first saved sample.
        elapsed = max(0, time.time() - self.start_time) if self.start_time is not None else 0
        self.limits.append((elapsed, low, high))


def generate_session_pdf(session):
    """Read the closed CSV and render a session summary using separate report figures.

    Runs in a background worker without updating live GUI widgets or reprocessing Host HR.

    Steps: (1) load the CSV columns into arrays, (2) take one HR value per packet
    (every 50th row), (3) page 1 = HR statistics + alarms + HR trend,
    (4) page 2 = last 10 s of PPG + its spectrum, (5) optional metadata pages,
    (6) write to a .tmp file and rename, so a crash never leaves a broken PDF.
    """
    # Read the completed recording; do not recalculate filtering or Host HR for the report.
    with open(session.filename, newline='') as source:
        rows = list(csv.DictReader(source))
    if not rows:
        return None
    times = np.array([float(row['Session Time (s)']) for row in rows])
    raw = np.array([float(row['Raw PPG (ADC)']) for row in rows])
    filtered = np.array([float(row['Filtered PPG']) for row in rows])
    host = np.array([float(row['Host HR (BPM)']) if row['Host HR (BPM)'] else np.nan for row in rows])
    mcu = np.array([float(row['MCU HR (BPM)']) if row['MCU HR (BPM)'] else np.nan for row in rows])
    # Per-packet HR statistics: each incoming estimate counts once, rather than
    # being weighted by its 50 duplicate sample rows.
    # Packets are exactly 50 rows in the existing CSV, including sequence wraps.
    packet_end = np.arange(49, len(rows), 50)
    valid_host = host[packet_end][np.isfinite(host[packet_end])]
    valid_mcu = mcu[packet_end][np.isfinite(mcu[packet_end])]
    paired = np.isfinite(host[packet_end]) & np.isfinite(mcu[packet_end])
    difference = host[packet_end][paired] - mcu[packet_end][paired]
    fmt = lambda value: '--' if value is None else f'{value:.1f}'
    mean = lambda values: float(np.mean(values)) if len(values) else None
    duration = max(0, times[-1] - times[0] + SAMPLE_PERIOD)
    pdf_path = os.path.splitext(session.filename)[0] + '.pdf'
    temporary_path = pdf_path + '.tmp'
    try:
        with PdfPages(temporary_path, metadata={'Title': 'LumiPulse Session Report', 'Author': 'LumiPulse'}) as pdf:
            blue, green, muted = '#253c51', '#39815c', '#596571'
            csv_name = os.path.basename(session.filename)

            # Link every report page to the source CSV and show its page number.
            def footer(figure, page):
                figure.text(.09, .035, textwrap.fill(csv_name, width=88),
                            fontsize=7, color=muted, va='top')
                figure.text(.94, .035, f'Page {page}', fontsize=8,
                            color=muted, ha='right', va='top')

            # Use consistent, light styling for report graphs.
            def style_axis(ax):
                ax.grid(True, alpha=.18)
                ax.tick_params(labelsize=9)
                ax.spines[['top', 'right']].set_visible(False)

            quality_text = '; '.join(f'{message}: {count}' for message, count in session.quality.items())
            quality_lines = textwrap.wrap(quality_text, width=88)
            seq = np.array([int(row['Packet Seq']) for row in rows])
            # Break only at sequence discontinuities; receipt jitter is not loss.
            hr_breaks = np.flatnonzero(np.diff(seq[packet_end]) % SEQ_MODULO != 1) + 1
            hr_time = times[packet_end]
            plot_time = np.insert(hr_time, hr_breaks, hr_time[hr_breaks])
            plot_host = np.insert(host[packet_end], hr_breaks, np.nan)
            plot_mcu = np.insert(mcu[packet_end], hr_breaks, np.nan)

            # PAGE 1: whole-session statistics, quality/alarm metadata and HR comparison.
            fig = Figure(figsize=(8.27, 11.69), facecolor='white')
            fig.text(.09, .945, 'LumiPulse', fontsize=25, weight='bold', color=blue)
            fig.text(.09, .910, 'PPG Recording Session Summary', fontsize=15, color=blue)
            fig.text(.09, .877, 'Recording start: ' + rows[0]['Timestamp'], fontsize=10, color=muted)
            fig.text(.09, .852, f'Estimated duration: {duration:.2f} s   |   Samples: {len(rows):,}   |   Sampling: {SAMPLE_RATE} Hz', fontsize=10)
            fig.text(.09, .827, textwrap.fill('CSV: ' + csv_name, width=88),
                     fontsize=8, color=muted, va='top')

            fig.text(.09, .762, 'HOST HEART RATE', fontsize=11, weight='bold', color=blue)
            for x_pos, label, value in (
                (.09, 'Mean', mean(valid_host)),
                (.38, 'Minimum', float(valid_host.min()) if len(valid_host) else None),
                (.67, 'Maximum', float(valid_host.max()) if len(valid_host) else None),
            ):
                fig.text(x_pos, .707, fmt(value), fontsize=27, weight='bold', color=green)
                fig.text(x_pos, .677, label + ' (BPM)', fontsize=10, color=muted)
            fig.text(.09, .636, f'MCU mean: {fmt(mean(valid_mcu))} BPM', fontsize=10)
            fig.text(.09, .613, f'Mean Host - MCU difference (paired): {fmt(mean(difference))} BPM', fontsize=10)

            fig.text(.09, .565, 'RECORDING PULSE SIGNAL & ALARMS', fontsize=11, weight='bold', color=blue)
            summary_lines = [
                f'Packets with available Host HR: {len(valid_host)} / {session.packet_count}   |   Quality-approved: {session.good_packets} / {session.packet_count}',
                f'Recorded signal coverage: {len(rows) / SAMPLE_RATE:.2f} s   |   Sequence discontinuities: {session.gaps}',
                f'Initial HR limits: {session.limits[0][1]:g} - {session.limits[0][2]:g} BPM',
                f'Final HR limits: {session.limits[-1][1]:g} - {session.limits[-1][2]:g} BPM   |   Limit changes: {len(session.limits) - 1}',
                f'Observed GUI HR alarm entries: Low {session.alarm_events["low"]}   |   High {session.alarm_events["high"]}',
                'Alarm entries reflect received available HR and the limits active at that time.',
            ]
            for index, line in enumerate(summary_lines):
                fig.text(.09, .540 - index * .022, line, fontsize=8.8,
                         color=muted if index == 5 else '#222222')
            for index, line in enumerate(quality_lines[:3]):
                fig.text(.09, .405 - index * .017, line, fontsize=8, color=muted)
            if len(quality_lines) > 3:
                fig.text(.09, .350, 'Extended quality and limit history follows the signal analysis.', fontsize=8, color=muted)

            hr_ax = fig.add_axes([.11, .095, .83, .225])
            hr_ax.plot(plot_time, plot_host, label='Host HR', lw=1, color=green)
            hr_ax.plot(plot_time, plot_mcu, '--', label='MCU HR', lw=1, color=blue)
            hr_ax.set_title('Full-session HR trend - Host / MCU', fontsize=11, loc='left', pad=10)
            hr_ax.set_ylabel('Heart rate (BPM)', fontsize=10)
            # Format units only; retain seconds in the data and automatic tick positions/axis limits.
            if duration < 120:
                hr_ax.set_xlabel('Session time (s)', fontsize=10)
            else:
                hr_ax.xaxis.set_major_formatter(
                    lambda value, position: f'{value / 60:.2f}'.rstrip('0').rstrip('.'))
                hr_ax.set_xlabel('Session time (min)', fontsize=10)
            hr_ax.legend(fontsize=8, loc='best')
            if not len(valid_host) and not len(valid_mcu):
                hr_ax.text(.5, .5, 'No HR estimates available', ha='center',
                           color=muted, transform=hr_ax.transAxes)
            style_axis(hr_ax)
            footer(fig, 1)
            pdf.savefig(fig)

            # PAGE 2: final continuous PPG excerpt and its spectrum, not whole-session waveforms.
            fig = Figure(figsize=(8.27, 11.69), facecolor='white')
            fig.text(.09, .945, 'LumiPulse', fontsize=25, weight='bold', color=blue)
            fig.text(.09, .910, 'Recorded Signal Analysis', fontsize=15, color=blue)
            axes = fig.subplots(2, 1)
            fig.subplots_adjust(left=.12, right=.94, top=.83, bottom=.16, hspace=.48)
            # Select the final continuous segment using sequence and timestamp
            # discontinuities. Never join samples from opposite sides of a gap.
            boundaries = np.flatnonzero((np.diff(times) > .1) | (np.diff(times) < 0) |
                ((np.diff(seq) != 0) & ((np.diff(seq) % 65536) != 1))) + 1
            segment_start = int(boundaries[-1]) if len(boundaries) else 0
            start = max(segment_start, len(rows) - 10 * SAMPLE_RATE)
            view_time = times[start:]
            fig.text(.09, .876, f'Final continuous segment: {len(view_time) / SAMPLE_RATE:.2f} s shown (maximum 10 s)',
                     fontsize=10, color=muted)
            axes[0].plot(view_time, raw[start:] - np.mean(raw[start:]), label='Raw PPG (DC removed)', lw=.8, color=blue)
            axes[0].plot(view_time, filtered[start:], label='Filtered PPG', lw=.8, color=green)
            axes[0].set_title('PPG waveform - final continuous portion', fontsize=12, loc='left', pad=12)
            axes[0].set_ylabel('Amplitude (ADC counts)', fontsize=10)
            axes[0].set_xlabel('Session time (s)', fontsize=10)
            axes[0].legend(fontsize=9)
            frequency, magnitude = spectral_magnitude(filtered[start:])
            axes[1].plot(frequency, magnitude, lw=1, color=blue)
            axes[1].set_xlim(0, 5)
            axes[1].set_title('Frequency spectrum - final filtered PPG segment', fontsize=12, loc='left', pad=12)
            axes[1].set_xlabel('Frequency (Hz)', fontsize=10)
            axes[1].set_ylabel('Spectral amplitude (ADC counts)', fontsize=10)
            if not len(frequency):
                axes[1].text(.5, .5, 'Insufficient continuous data (need 2 s)', ha='center', transform=axes[1].transAxes)
            # PDF amplitude retains its separate three-second quality-approved rule.
            amplitude = (float(np.ptp(filtered[-150:]))
                         if len(rows) - segment_start >= 150 and session.valid_samples >= 150 else None)
            fig.text(.09, .085, f'Recent filtered peak-to-peak (3 s, quality-approved): {fmt(amplitude)} ADC counts', fontsize=9)
            footer(fig, 2)
            for ax in axes:
                style_axis(ax)
            pdf.savefig(fig)
            # OPTIONAL PAGES: keep extended limit history and quality details out of the main summary.
            if len(session.limits) > 1 or len(quality_lines) > 3:
                details = ['HR limit history (session seconds / low / high BPM)']
                details += [f'{elapsed:.2f} s   {low:g} - {high:g}'
                            for elapsed, low, high in session.limits]
                details += ['', 'Pulse-signal check, packet counts'] + quality_lines
                for offset in range(0, len(details), 38):
                    fig = Figure(figsize=(8.27, 11.69), facecolor='white')
                    fig.text(.09, .94, 'Session metadata', fontsize=18, weight='bold')
                    for index, line in enumerate(details[offset:offset + 38]):
                        fig.text(.09, .88 - index * .021, line, fontsize=10)
                    footer(fig, 3 + offset // 38)
                    pdf.savefig(fig)
        # Publish only a completed PDF; an error leaves the source CSV intact.
        os.replace(temporary_path, pdf_path)
    except Exception:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise
    return pdf_path


class SessionReports:
    """Run PDF jobs in the background and report completion through the GUI thread.

    Building a PDF takes a moment, so it runs on a worker thread to keep the GUI
    responsive; main() calls poll() every loop to log the result when it finishes.
    """
    def __init__(self):
        # Generate one report at a time, using separate figures from the live GUI.
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='LumiPulsePDF')
        self.pending = []
        self.submitted = set()

    def submit(self, session):
        # Submit each nonempty completed recording at most once.
        if session is None or not session.samples or session.filename in self.submitted:
            return
        self.submitted.add(session.filename)
        self.pending.append(self.worker.submit(generate_session_pdf, session))

    def poll(self, window):
        # Check finished jobs without waiting; log their results on the GUI thread.
        for future in self.pending[:]:
            if not future.done():
                continue
            self.pending.remove(future)
            try:
                filename = future.result()
                if filename:
                    add_log(window, 'PDF saved: ' + os.path.abspath(filename))
            except Exception as error:
                add_log(window, f'PDF generation failed (CSV retained): {error}')

    def close(self, window):
        # Normal exit waits for pending reports before destroying the window.
        self.worker.shutdown(wait=True)
        self.poll(window)

# ============================================================
# SYSTEM LOG
# ============================================================

def add_log(window, message):
    """Add a system-log event using the required time.ctime() format (Reqt 13).

    Example: Thu Sep 19 17:46:50 2024: Pulse Low
    """
    add_log_line(window, f"{time.ctime()}: {message}")


def add_log_line(window, line):
    """Append an already timestamped event, such as a message from PPGLink.

    Author: Bianca Dibden (integration)
    """
    window["-SYSTEM_LOG-"].update(line + "\n", append=True)

# ============================================================
# COMMUNICATION STATUS
# ============================================================
# Packet validation, the 5 s no-data timer, reconnection and the
# matching log messages are handled in comms.py (PPGLink). The GUI
# only displays the state that PPGLink reports.

def register_valid_packet(window, record):
    """Update the hidden last-packet field for a validated HostRecord."""
    packet_time = datetime.fromtimestamp(record.time).strftime("%H:%M:%S")
    window["-LAST_PACKET-"].update(f"{packet_time}  (#{record.seq})")


def update_communication_display(window, communication, link):
    """Show link state and the no-data alarm separately (Reqt 14).

    Link state (BT status): CONNECTED / DISCONNECTED - RECONNECTING / DISCONNECTED.
    Data state (red banner): DEVICE DISCONNECTED, or NO DATA FOR n s while the
    link is open but no valid packet has arrived for 5 s or more.
    Host HR is cleared whenever the link is down or the alarm is active.
    Author: Bianca Dibden (integration)
    """
    connected = link.status == STATUS_CONNECTED
    alarm = link.no_data_alarm      # derived from the clock: read once per poll
    # Never leave a stale HR on screen while there is no link or no data.
    # Fresh data resumes the HR display once the link is back and the alarm clears.
    if not connected or alarm:
        window["-CURRENT_HR-"].update("--.-", text_color=TEXT_MUTED)
        message = "DEVICE DISCONNECTED" if not connected else "NO DATA - CHECK DEVICE"
        window["-HR_STATUS-"].update(message, text_color=RED)
        # Same "Pulse signal ..." wording as ppg_dsp.SignalQuality.message().
        if not connected:
            pulse_state, pulse_detail = "\u25cf NO DEVICE", "Device disconnected - no pulse signal"
        else:
            pulse_state, pulse_detail = "\u25cf NO DATA", "No pulse data received - check the device"
        window["-PULSE_SIGNAL_STATE-"].update(pulse_state, text_color=TEXT_MUTED)
        window["-PULSE_SIGNAL_DETAIL-"].update(pulse_detail, text_color=TEXT_MUTED)
        window["-SIGNAL_QUALITY-"].update("Pulse signal unavailable (no data)", text_color=TEXT_MUTED)

    # Skip unchanged text; the seconds count only matters while the alarm is active.
    silent_s = int(link.seconds_since_packet) if alarm else 0
    state = (link.status, alarm, silent_s)
    if state == communication["shown"]:
        return
    communication["shown"] = state

    if connected and not alarm:
        window["-BT_STATUS-"].update("\u25cf CONNECTED", text_color=GREEN)
    elif connected:
        window["-BT_STATUS-"].update("\u25cf CONNECTED - NO DATA", text_color=AMBER)
    elif link.status == STATUS_CONNECTING:
        # A connect attempt can block for 10+ s on a dead Bluetooth port.
        window["-BT_STATUS-"].update("\u25cf DISCONNECTED - RECONNECTING...", text_color=AMBER)
    else:
        window["-BT_STATUS-"].update("\u25cf DISCONNECTED", text_color=RED)

    if not connected:
        banner = "\u26a0 DEVICE DISCONNECTED"
        if alarm:
            banner += f" - NO DATA FOR {silent_s} s"
    elif alarm:
        banner = f"\u26a0 NO DATA FOR {silent_s} s (link open)"
    else:
        banner = ""
    window["-COMM_ALARM-"].update(banner)

# ============================================================
# CSV RECORDING
# ============================================================

def start_recording():
    """Create a uniquely named recording CSV and write its column headings."""
    # Save under the launch directory; create a new file rather than overwrite an earlier one.
    os.makedirs("LumiPulse_Recordings", exist_ok=True)
    file_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join("LumiPulse_Recordings", f"LumiPulse_{file_time}.csv")

    base, suffix = os.path.splitext(filename)
    counter = 1
    while os.path.exists(filename):
        filename = f"{base}_{counter}{suffix}"
        counter += 1
    csv_file = open(filename, "x", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "Timestamp",
        "Session Time (s)",
        "Packet Seq",
        "Raw PPG (ADC)",
        "Filtered PPG",
        "MCU HR (BPM)",
        "Host HR (BPM)",
        "Recording State"
    ])
    return csv_file, csv_writer, filename


def save_packet(csv_writer, record, recording_start_time):
    """Write one CSV row per sample, with unavailable HR values left blank.

    Estimate sample times at 20 ms spacing, ending at record.time (host time after DSP).
    These are reconstructed times, not measured MCU acquisition timestamps.
    Author: Bianca Dibden (integration)
    """
    n = len(record.raw)
    fmt = lambda v: "" if v is None else f"{v:.1f}"
    for i in range(n):
        t = record.time - (n - 1 - i) * SAMPLE_PERIOD  # Count back in 20 ms steps from host record time.
        timestamp = datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # Local time, to milliseconds.
        csv_writer.writerow([
            timestamp,
            f"{t - recording_start_time:.3f}",
            record.seq,
            int(record.raw[i]),
            f"{record.filtered[i]:.2f}",
            fmt(record.mcu_hr),
            fmt(record.host_hr),
            "ON"
        ])


def stop_recording(csv_file):
    """Close and flush the CSV if a recording file is open."""
    if csv_file is not None:
        csv_file.close()

# ============================================================
# HEART RATE STATUS
# ============================================================

def update_hr_status(
    window,
    heart_rate,
    low_limit,
    high_limit
):
    """Colour the Host HR display and status according to the active low/high limits.

    Red = outside limits (alarm), green = within limits.
    """

    if heart_rate < low_limit:
        window["-HR_STATUS-"].update(
            "⚠ LOW HEART RATE",
            text_color=RED
        )
        window["-CURRENT_HR-"].update(
            text_color=RED
        )

    elif heart_rate > high_limit:
        window["-HR_STATUS-"].update(
            "⚠ HIGH HEART RATE",
            text_color=RED
        )
        window["-CURRENT_HR-"].update(
            text_color=RED
        )

    else:
        window["-HR_STATUS-"].update(
            "✓ HEART RATE NORMAL",
            text_color=GREEN
        )
        window["-CURRENT_HR-"].update(
            text_color=GREEN
        )

# ============================================================
# GUI LAYOUT AND RESIZING
# ============================================================
# FreeSimpleGUI stacks every layout row with Tk's "pack". These configure_*()
# functions run AFTER the window exists and re-arrange the Tk widgets into
# grids with fixed proportions, so panels and fonts scale when the window is
# resized. They are cosmetic: no data handling happens here.

def configure_monitor_grid(window):
    """Allocate the Monitor layout once; Tk handles later changes in window size.

    The plot heights split 1:1 and the log sits below both. The right-hand column (HR box,
    limits, pulse signal, session) is one cell; configure_right_column() gives any spare
    height to the HR box. The main column widths use a 65:35 ratio.
    """
    cells = (
        ("-TREND_PANEL-", 0, 0, 3, 1),
        ("-RIGHT_COLUMN-", 0, 1, 6, 1),
        ("-PPG_PANEL-", 3, 0, 3, 1),
        ("-LOG_PANEL-", 6, 0, 1, 2),
    )
    wrappers = [window[key].ParentRowFrame for key, *_ in cells]
    monitor = wrappers[0].master
    # Remove these rows from pack before using grid in the same parent container.
    for wrapper in wrappers:
        wrapper.pack_forget()
    for (key, row, column, rowspan, colspan), wrapper in zip(cells, wrappers):
        wrapper.grid(row=row, column=column, rowspan=rowspan,
                     columnspan=colspan, sticky="nsew")
        window[key].Widget.pack_configure(fill="both", expand=True)

    for column, weight in ((0, 13), (1, 7)):
        monitor.grid_columnconfigure(column, weight=weight, uniform="monitor_columns")
    for row in range(6):
        monitor.grid_rowconfigure(row, weight=1, uniform="monitor_main_rows")
    monitor.grid_rowconfigure(6, weight=0)

    # Let the plot canvas fill its panel, rather than resizing only the panel border.
    for key in ("-HR_TREND_CANVAS-", "-PPG_CANVAS-"):
        window[key].ParentRowFrame.pack_configure(fill="both", expand=True)


def configure_right_column(window):
    """Give spare height in the right-hand column to the HR box; keep the rest compact."""
    for key, grows in (("-HR_PANEL-", True), ("-HR_CONTROLS_PANEL-", False),
                       ("-PULSE_PANEL-", False), ("-SESSION_PANEL-", False)):
        row = window[key].ParentRowFrame
        if grows:
            row.pack_configure(fill="both", expand=True)
        else:
            row.pack_configure(fill="x", expand=False)


def configure_host_hr_panel(window):
    """Centre the Host HR number in its own box and scale it up as the box grows."""
    window.TKroot.update_idletasks()
    panel = window["-HR_PANEL-"].Widget
    number = window["-CURRENT_HR-"].Widget
    bpm = window["-HR_BPM_LABEL-"].Widget
    number_row = window["-CURRENT_HR-"].ParentRowFrame
    heading_row = window["-HR_HEADING-"].ParentRowFrame
    # The number/BPM row takes all spare space and sits in the middle of it.
    number_row.pack_configure(expand=True, fill="none", anchor="center")
    window.TKroot.update_idletasks()
    base_row_width = number_row.winfo_reqwidth()
    base_number_height = int(number.tk.call("font", "metrics", number.cget("font"), "-linespace"))
    # Keep the box's requested size fixed so larger fonts cannot trigger a resize loop.
    panel.configure(width=panel.winfo_reqwidth(), height=panel.winfo_reqheight())
    panel.pack_propagate(False)
    pending = None
    last_style = (FONT_HR[1], FONT_VALUE[1])

    def apply_scale():
        nonlocal pending, last_style
        pending = None
        spare_width = max(1, panel.winfo_width() - 30)
        spare_height = max(1, panel.winfo_height() - heading_row.winfo_height() - 24)
        # Fit the number within the box; never below the designed size, capped for sanity.
        scale = max(1.0, min(2.4, spare_width / base_row_width,
                             spare_height / base_number_height))
        style = (round(FONT_HR[1] * scale), round(FONT_VALUE[1] * scale))
        if style == last_style:
            return
        number.configure(font=(FONT_HR[0], style[0], *FONT_HR[2:]))
        bpm.configure(font=(FONT_VALUE[0], style[1], *FONT_VALUE[2:]))
        last_style = style

    def schedule_scale(event):
        nonlocal pending
        # Combine repeated resize events into one update when Tk is idle.
        if pending is None:
            pending = panel.after_idle(apply_scale)

    def cancel_scale(event):
        nonlocal pending
        # Cancel a pending update if this panel is being destroyed.
        if event.widget is panel and pending is not None:
            panel.after_cancel(pending)
            pending = None

    panel.bind("<Configure>", schedule_scale, add="+")
    panel.bind("<Destroy>", cancel_scale, add="+")


def configure_main_tabs(window):
    """Scale the Monitor/Researcher tab labels and padding up slightly as the window grows.

    The tabs keep their normal size at the starting window width and grow by about half
    the window's relative growth (capped), so they stay proportionate but easy to hit.
    """
    from tkinter import font as tkfont, ttk
    root = window.TKroot
    root.update_idletasks()
    notebook = window["-MAIN_TABS-"].Widget
    tab_style = f"{notebook.cget('style')}.Tab"
    style = ttk.Style(root)
    base_font = tkfont.nametofont("TkDefaultFont").actual()
    base_size = abs(int(base_font["size"])) + 1   # 1 pt larger, like the rest of the GUI text
    base_family = base_font["family"]
    base_pad = (12, 5)              # (horizontal, vertical) padding at the starting size
    base_width = max(1, root.winfo_width())
    pending = None
    last = None

    def apply_scale():
        nonlocal pending, last
        pending = None
        growth = max(0.0, root.winfo_width() / base_width - 1.0)
        scale = min(1.5, 1.0 + 0.5 * growth)
        values = (round(base_size * scale), round(base_pad[0] * scale), round(base_pad[1] * scale))
        if values == last:
            return
        last = values
        style.configure(tab_style, font=(base_family, values[0], "bold"),
                        padding=(values[1], values[2]))

    def schedule_scale(event):
        nonlocal pending
        # Only react to the window itself, and combine bursts of resize events.
        if event.widget is root and pending is None:
            pending = root.after_idle(apply_scale)

    root.bind("<Configure>", schedule_scale, add="+")
    apply_scale()


def configure_session_timer(window):
    """Scale only the Session timer font, keeping it within the available panel space."""
    from tkinter import font as tkfont

    window.TKroot.update_idletasks()
    panel = window["-SESSION_PANEL-"].Widget
    timer = window["-SESSION_TIME-"].Widget
    timer_row = window["-SESSION_TIME-"].ParentRowFrame
    base_width, base_height = panel.winfo_width(), panel.winfo_height()
    timer_font = tkfont.Font(root=window.TKroot, font=timer.cget("font"))
    base_size = timer_font.actual("size")
    # Reserve width for a long timer value before allowing growth up to 24 pt.
    text_width = timer_font.measure("99:59:59")
    other_width = (timer_row.winfo_reqwidth() - timer_font.measure(timer.cget("text"))
                   + 20)
    timer_height = timer.winfo_reqheight()
    timer_top = timer.winfo_rooty() - panel.winfo_rooty()
    panel.configure(width=panel.winfo_reqwidth(), height=panel.winfo_reqheight())
    panel.pack_propagate(False)
    timer.configure(font=timer_font)
    pending = None
    last_size = base_size

    def apply_scale():
        nonlocal pending, last_size
        pending = None
        scale = max(1.0, min(1.20, panel.winfo_width() / base_width,
                             panel.winfo_height() / base_height,
                             max(0, panel.winfo_width() - other_width) / text_width,
                             max(0, panel.winfo_height() - timer_top - 10) / timer_height))
        size = min(24, round(base_size * scale))
        if size != last_size:
            timer_font.configure(size=size)
            last_size = size

    def schedule_scale(event):
        nonlocal pending
        # Combine repeated resize events into one update when Tk is idle.
        if pending is None:
            pending = panel.after_idle(apply_scale)

    def cancel_scale(event):
        nonlocal pending
        # Cancel a pending update if this panel is being destroyed.
        if event.widget is panel and pending is not None:
            panel.after_cancel(pending)
            pending = None

    panel.bind("<Configure>", schedule_scale, add="+")
    panel.bind("<Destroy>", cancel_scale, add="+")


def configure_researcher_grid(window):
    """Allocate 78% of Researcher width to plots and 22% to the information sidebar."""
    canvas_keys = ("-HR_COMPARISON_CANVAS-", "-FFT_CANVAS-",
                   "-SPECTROGRAM_CANVAS-", "-DERIVATIVE_CANVAS-")
    panels = [window[key].ParentRowFrame.master for key in canvas_keys]
    wrappers = [panel.master for panel in panels]
    sidebar = window["-RESEARCH_SIDEBAR-"]
    sidebar_row = sidebar.ParentRowFrame
    researcher = wrappers[0].master
    headings = [row for row in researcher.pack_slaves()
                if row not in wrappers and row != sidebar_row]
    for row in headings + wrappers + [sidebar_row]:
        row.pack_forget()
    for index, row in enumerate(headings):
        row.grid(row=index, column=0, columnspan=3, sticky="ew")
    for panel, wrapper, (row, column) in zip(panels, wrappers,
                                            ((2, 0), (2, 1), (3, 0), (3, 1))):
        wrapper.grid(row=row, column=column, sticky="nsew")
        panel.pack_configure(fill="both", expand=True, padx=4, pady=4)
    sidebar_row.grid(row=2, column=2, rowspan=2, sticky="nsew")
    sidebar.Widget.pack_configure(fill="both", expand=True, padx=4, pady=4)
    for column, weight in enumerate((39, 39, 22)):
        researcher.grid_columnconfigure(column, weight=weight, uniform="researcher_columns")
    for row in (0, 1):
        researcher.grid_rowconfigure(row, weight=0)
    for row in (2, 3):
        researcher.grid_rowconfigure(row, weight=1, uniform="researcher_plots")
    for key in canvas_keys:
        window[key].ParentRowFrame.pack_configure(fill="both", expand=True)
    # Let Processing fill spare sidebar height; keep the other information panels compact.
    processing = window["-PROCESSING_TABS-"]
    processing_frame = processing.ParentRowFrame.master
    processing_frame.master.pack_configure(fill="both", expand=True)
    processing_frame.pack_configure(fill="both", expand=True)
    processing.ParentRowFrame.pack_configure(fill="both", expand=True)


def fit_initial_window(window):
    """Fit and centre the initial window on screen, then let the existing grids resize it."""
    root = window.TKroot
    root.update_idletasks()
    wm_width, wm_height = root.maxsize()
    available_width = max(1, min(root.winfo_vrootwidth(), wm_width) - 32)
    available_height = max(1, min(root.winfo_vrootheight(), wm_height) - 32)

    # Plot canvases have comfortable requested sizes, not hard minimums.
    # Base the horizontal minimum on the actual right-side controls instead.
    monitor_width = max(window[key].Widget.winfo_reqwidth()
                        for key in ("-HR_CONTROLS_PANEL-", "-SESSION_PANEL-")) / .35
    researcher_width = window["-RESEARCH_SIDEBAR-"].Widget.winfo_reqwidth() / .22
    minimum_width = min(available_width, int(max(monitor_width, researcher_width) + 50))
    minimum_height = min(available_height, root.winfo_reqheight())
    root.minsize(minimum_width, minimum_height)

    width = min(available_width, max(minimum_width, root.winfo_reqwidth()))
    height = min(available_height, max(minimum_height, root.winfo_reqheight()))
    left = root.winfo_vrootx() + 16 + (available_width - width) // 2
    top = root.winfo_vrooty() + 32 + (available_height - height) // 2
    root.geometry(f"{width}x{height}+{left}+{top}")


# ============================================================
# GUI CONSTRUCTION
# ============================================================
def create_gui():
    """Build both tabs, finalise the widgets and configure their layout and resizing.

    Layout is a list of rows; each row is a list of elements. The pattern used
    throughout: define each panel's layout, wrap it in an sg.Frame, then combine
    the frames into the Monitor and Researcher tabs inside one sg.Window.
      Monitor tab:    HR trend, Host HR + limits, live PPG, session info, system log
      Researcher tab: MCU-vs-Host HR, FFT, spectrogram, 2nd derivative, info sidebar
    """

    sg.theme("DarkGrey13")
    sg.set_options(
        background_color=BG,
        text_color=TEXT,
        element_background_color=BG,
        input_elements_background_color=PANEL,
        input_text_color=TEXT,
        button_color=(TEXT, PANEL),
        font=FONT_NORMAL
    )

    # ========================================================
    # HEADER
    # ========================================================

    header = [
        sg.Text(
            "LUMIPULSE",
            font=FONT_TITLE,
            text_color=ACCENT,
            background_color=BG,
            expand_x=True
        ),
        sg.Text(
            "Bluetooth:",
            text_color=TEXT_MUTED,
            background_color=BG
        ),
        sg.Text(
            "● DISCONNECTED",
            key="-BT_STATUS-",
            text_color=RED,
            background_color=BG
        ),
        sg.Text(
            "",
            key="-COMM_ALARM-",
            font=("Helvetica", 11, "bold"),
            text_color=RED,
            background_color=BG
        ),
    ]

    # ========================================================
    # HOST HR DISPLAY
    # ========================================================

    current_hr_layout = [
        [
            sg.Text(
                "HOST HEART RATE",
                key="-HR_HEADING-",
                font=FONT_SECTION,
                text_color=TEXT_MUTED,
                background_color=PANEL
            )
        ],
        [
            sg.Text(
                "--.-",
                key="-CURRENT_HR-",
                size=(5, 1),
                justification="right",
                pad=((0, 2), 0),
                font=FONT_HR,
                text_color=GREEN,
                background_color=PANEL
            ),
            sg.Text(
                "BPM",
                key="-HR_BPM_LABEL-",
                pad=((2, 0), 0),
                font=FONT_VALUE,
                text_color=TEXT_MUTED,
                background_color=PANEL
            )
        ]
    ]

    # ========================================================
    # HR LIMITS PANEL
    # ========================================================

    hr_limits_layout = [
        [
            sg.Text(
                "HR LIMITS",
                key="-HR_LIMITS_HEADING-",
                font=FONT_SECTION,
                text_color=TEXT_MUTED,
                background_color=PANEL
            )
        ],
        [
            sg.Text(
                "Low",
                font=("Helvetica", 14),
                size=(7, 1),
                background_color=PANEL
            ),
            sg.Input(
                str(LOW_HR_LIMIT),
                font=("Helvetica", 14),
                size=(7, 1),
                key="-LOW_HR_LIMIT-",
                enable_events=True
            ),
            sg.Text(
                "BPM",
                font=("Helvetica", 14),
                text_color=TEXT_MUTED,
                background_color=PANEL
            )
        ],
        [
            sg.Text(
                "High",
                font=("Helvetica", 14),
                size=(7, 1),
                background_color=PANEL
            ),
            sg.Input(
                str(HIGH_HR_LIMIT),
                font=("Helvetica", 14),
                size=(7, 1),
                key="-HIGH_HR_LIMIT-",
                enable_events=True
            ),
            sg.Text(
                "BPM",
                font=("Helvetica", 14),
                text_color=TEXT_MUTED,
                background_color=PANEL
            )
        ],
        [
            sg.Button(
                "Apply Limits",
                font=("Helvetica", 14),
                key="-APPLY_LIMITS-",
                size=(12, 1)
            ),
            sg.Text(
                "",
                key="-HR_LIMIT_ERROR-",
                size=(17, 1),
                font=FONT_SMALL,
                text_color=RED,
                background_color=PANEL,
                pad=((4, 0), 0)
            )
        ],
        [
            sg.Text(
                "WAITING FOR HEART RATE",
                key="-HR_STATUS-",
                font=("Helvetica", 12, "bold"),
                size=(28, 2),
                text_color=AMBER,
                background_color=PANEL
            )
        ],
        # Separate box for the PPG pulse-signal check (ppg_dsp.py). Deliberately not
        # called "signal quality" so it is not confused with the Bluetooth link above.
        [
            sg.Frame(
                "PULSE SIGNAL",
                key="-PULSE_PANEL-",
                layout=[
                    [
                        sg.Text(
                            "\u25cf WAITING",
                            key="-PULSE_SIGNAL_STATE-",
                            font=("Helvetica", 12, "bold"),
                            size=(20, 1),  # wide enough for "● CHECK SENSOR" in bold
                            text_color=TEXT_MUTED,
                            background_color=PANEL
                        )
                    ],
                    [
                        sg.Text(
                            "Waiting for pulse data",
                            key="-PULSE_SIGNAL_DETAIL-",
                            font=FONT_SMALL,
                            size=(34, 4),  # up to 4 wrapped lines for long "check sensor" reasons
                            text_color=TEXT_MUTED,
                            background_color=PANEL
                        )
                    ]
                ],
                font=FONT_SMALL,
                title_color=TEXT_MUTED,
                background_color=PANEL,
                border_width=1,
                expand_x=True
            )
        ]
    ]

    # ========================================================
    # COMBINE HOST HR AND LIMIT CONTROLS
    # ========================================================

    # One panel: large HR/BPM, then limits, Apply, and textual alarm/status.
    # The last row of hr_limits_layout is the PULSE SIGNAL frame; it gets its own box.
    pulse_signal_row = [hr_limits_layout.pop()[0]]

    # ========================================================
    # HEART RATE TREND PANEL
    # ========================================================

    hr_trend_layout = [
        [
            sg.Text(
                "HOST HEART RATE TREND",
                font=FONT_SECTION,
                text_color=TEXT_MUTED,
                background_color=PANEL
            )
        ],
        [
            sg.Canvas(
                key="-HR_TREND_CANVAS-",
                size=(420, 160),
                background_color=PANEL,
                expand_x=True,
                expand_y=True
            )
        ]
    ]

    # ========================================================
    # PPG PANEL
    # ========================================================

    ppg_layout = [
        [
            sg.Text(
                "LIVE PPG WAVEFORM",
                font=FONT_SECTION,
                text_color=TEXT_MUTED,
                background_color=PANEL
            )
        ],
        [
            sg.Canvas(
                key="-PPG_CANVAS-",
                size=(420, 160),
                background_color=PANEL,
                expand_x=True,
                expand_y=True
            )
        ],
        [
            sg.Text(
                "Signal display:",
                text_color=TEXT_MUTED,
                background_color=PANEL
            ),
            sg.Radio(
                "Raw",
                group_id="PPG_DISPLAY",
                default=True,
                enable_events=True,
                key="-SHOW_RAW-",
                background_color=PANEL,
                text_color=TEXT
            ),
            sg.Radio(
                "Filtered",
                group_id="PPG_DISPLAY",
                enable_events=True,
                key="-SHOW_FILTERED-",
                background_color=PANEL,
                text_color=TEXT
            ),
            sg.Radio(
                "Both",
                group_id="PPG_DISPLAY",
                enable_events=True,
                key="-SHOW_BOTH-",
                background_color=PANEL,
                text_color=TEXT
            )
        ]
    ]

    # ========================================================
    # SESSION PANEL
    # ========================================================

    session_layout = [
        [
            sg.Text(
                "SESSION",
                font=FONT_SECTION,
                text_color=TEXT_MUTED,
                background_color=PANEL
            )
        ],
        [
            sg.Text(
                "Recording",
                text_color=TEXT_MUTED,
                background_color=PANEL
            )
        ],
        [
            sg.Text(
                "● OFF",
                key="-SESSION_RECORDING-",
                font=FONT_VALUE,
                text_color=TEXT_MUTED,
                background_color=PANEL
            )
        ],
        [
            sg.HorizontalSeparator()
        ],
        [
            sg.Text(
                "Elapsed time",
                text_color=TEXT_MUTED,
                background_color=PANEL
            )
        ],
        [
            sg.Text(
                "00:00:00",
                key="-SESSION_TIME-",
                font=("Helvetica", 21, "bold"),
                text_color=TEXT,
                background_color=PANEL
            )
        ]
    ]

    # ========================================================
    # SYSTEM LOG
    # ========================================================

    system_log_layout = [
        [
            sg.Text(
                "SYSTEM LOG",
                font=FONT_SECTION,
                text_color=TEXT_MUTED,
                background_color=PANEL
            )
        ],
        [
            sg.Multiline(
                size=(60, 4),
                key="-SYSTEM_LOG-",
                disabled=True,
                autoscroll=True,
                expand_x=True,
                background_color=BG,
                text_color=TEXT
            )
        ]
    ]

    # ========================================================
    # MONITOR TAB
    # ========================================================
    # Each panel starts in its own row; configure_monitor_grid() arranges these rows in a grid.
    monitor_layout = [
        [sg.Frame(
            "", hr_trend_layout, key="-TREND_PANEL-",
            background_color=PANEL, border_width=1,
            expand_x=True, expand_y=True
        )],
        [sg.Frame(
            "", ppg_layout, key="-PPG_PANEL-",
            background_color=PANEL, border_width=1,
            expand_x=True, expand_y=True
        )],
        [sg.Column([
            [sg.Frame(
                "", current_hr_layout, key="-HR_PANEL-",
                background_color=PANEL, border_width=1,
                expand_x=True, expand_y=True
            )],
            [sg.Frame(
                "", hr_limits_layout, key="-HR_CONTROLS_PANEL-",
                background_color=PANEL, border_width=1,
                expand_x=True
            )],
            pulse_signal_row,
            [sg.Frame(
                "", session_layout, key="-SESSION_PANEL-",
                background_color=PANEL, border_width=1,
                expand_x=True
            )]
        ], key="-RIGHT_COLUMN-", background_color=BG, pad=(0, 0),
           expand_x=True, expand_y=True, vertical_alignment="top")],
        [sg.Frame(
            "", system_log_layout, key="-LOG_PANEL-",
            background_color=PANEL, border_width=1,
            expand_x=True
        )]
    ]

    # ========================================================
    # RESEARCHER - MCU VS HOST HR
    # ========================================================

    researcher_hr_layout = [
        [
            sg.Text(
                "MCU vs HOST HEART RATE",
                font=FONT_SECTION,
                text_color=TEXT_MUTED,
                background_color=PANEL
            )
        ],
        [
            sg.Text(
                "MCU HR",
                size=(14, 1),
                text_color=TEXT_MUTED,
                background_color=PANEL
            ),
            sg.Text(
                "--.- BPM",
                key="-MCU_HR-",
                font=FONT_VALUE,
                text_color=ACCENT,
                background_color=PANEL
            )
        ],
        [
            sg.Text(
                "Host HR",
                size=(14, 1),
                text_color=TEXT_MUTED,
                background_color=PANEL
            ),
            sg.Text(
                "--.- BPM",
                key="-HOST_HR-",
                font=FONT_VALUE,
                text_color=GREEN,
                background_color=PANEL
            )
        ],
        [
            sg.Text(
                "Difference",
                size=(14, 1),
                text_color=TEXT_MUTED,
                background_color=PANEL
            ),
            sg.Text(
                "--.- BPM",
                key="-HR_DIFFERENCE-",
                background_color=PANEL
            )
        ]
    ]

    # ========================================================
    # RESEARCHER - HR COMPARISON GRAPH
    # ========================================================

    comparison_plot_layout = [
        [
            sg.Text(
                "MCU vs HOST HR TREND",
                font=FONT_SECTION,
                text_color=TEXT_MUTED,
                background_color=PANEL
            )
        ],
        [
            sg.Canvas(
                key="-HR_COMPARISON_CANVAS-",
                size=(420, 150),
                expand_x=True, expand_y=True,
                background_color=PANEL
            )
        ]
    ]

    # ========================================================
    # RESEARCHER - ADDITIONAL SIGNAL ANALYSIS
    # ========================================================

    signal_analysis_layout = [
        [
            sg.Text(
                "ADDITIONAL SIGNAL ANALYSIS",
                font=FONT_SECTION,
                text_color=TEXT_MUTED,
                background_color=PANEL
            )
        ],
        [
            sg.Text(
                "Pulse signal:",
                size=(18, 1),
                text_color=TEXT_MUTED,
                background_color=PANEL
            ),
            sg.Text(
                "--",
                key="-SIGNAL_QUALITY-",
                size=(34, 5),
                justification="left",
                background_color=PANEL
            )
        ],
        [
            sg.Text(
                "Detected pulses:",
                size=(18, 1),
                text_color=TEXT_MUTED,
                background_color=PANEL
            ),
            sg.Text(
                "--",
                key="-PULSE_COUNT-",
                background_color=PANEL
            )
        ]
    ]

    # Reuse the existing measurement elements/keys in the final sidebar.
    # The original pulse-count row is not displayed.
    researcher_hr_layout[0][0].DisplayText = "HEART RATE"
    signal_analysis_layout[0][0].DisplayText = "SIGNAL SUMMARY"
    quality_row = signal_analysis_layout[1]
    quality_value = quality_row[1]
    pulse_value = sg.Text("--", key="-PULSE_COUNT-", visible=False)

    # Keep hidden elements so existing packet/status updates still find their keys.
    filter_value = sg.Text("Waiting for data", key="-FILTER_STATUS-", visible=False)
    last_packet_value = sg.Text("--", key="-LAST_PACKET-", visible=False)
    # PROCESSING SUBTABS: matching label/value heights keeps their first lines aligned.
    def processing_rows(entries):
        return [[
            sg.Text(label, size=(9, lines), font=FONT_NORMAL, text_color=TEXT_MUTED,
                    background_color=PANEL, pad=(2, 2)),
            sg.Text(value, size=(17, lines), font=FONT_NORMAL, text_color=TEXT,
                    background_color=PANEL, pad=(2, 2))
        ] for label, value, lines in entries]

    host_processing_rows = processing_rows([
        ("Sampling:", "50 Hz (from MCU)", 2),
        ("Filter:", "0.5-5 Hz band-pass", 2),
        ("Detection:", "Adaptive threshold", 2),
        ("HR:", "Median of 3-6 IBIs", 2),
    ])

    processing_layout = [
        [sg.Text("PROCESSING", font=FONT_SECTION, text_color=TEXT_MUTED, background_color=PANEL)],
        [sg.TabGroup([[
            sg.Tab("MCU", processing_rows([
                ("Sampling:", "50 Hz", 1),
                ("Filter:", "1st-order band-pass, 0.5-5 Hz", 2),
                ("Detection:", "Adaptive threshold (P10/P90)", 2),
            ]), background_color=PANEL),
            sg.Tab("HOST", host_processing_rows, background_color=PANEL),
        ]], key="-PROCESSING_TABS-", size=(1, 110), font=FONT_NORMAL,
            background_color=PANEL, title_color=TEXT_MUTED,
            selected_title_color=TEXT, selected_background_color=PANEL,
            tab_background_color=BG, expand_x=True, expand_y=True, pad=(2, 2))],
        [filter_value, last_packet_value, pulse_value]
    ]
    # Each measurement is a muted label with its value directly beneath; every text
    # is left-justified with the same padding so the labels and values share one
    # left edge, in line with the pulse-signal text above.
    def summary_label(text, key):
        return [sg.Text(text, key=key, justification="left",
                        text_color=TEXT_MUTED, background_color=PANEL)]

    signal_analysis_layout = [
        signal_analysis_layout[0],
        [quality_row[0]], [quality_value],
        summary_label("Average Host HR:", "-AVERAGE_HR_LABEL-"),
        [sg.Text("--", key="-AVERAGE_HR-", justification="left", background_color=PANEL)],
        summary_label("PPG amplitude (P-P):", "-AMPLITUDE_LABEL-"),
        [sg.Text("--", key="-AMPLITUDE-", justification="left", background_color=PANEL)],
    ]
    # Reserve space for changing values so live updates do not move the panels.
    value_sizes = {
        "-MCU_HR-": (10, 1), "-HOST_HR-": (10, 1),
        "-HR_DIFFERENCE-": (10, 1), "-FILTER_STATUS-": (24, 2),
        "-LAST_PACKET-": (24, 1), "-SIGNAL_QUALITY-": (26, 5),
        "-PULSE_COUNT-": (4, 1),
        "-AVERAGE_HR_LABEL-": (20, 1), "-AMPLITUDE_LABEL-": (20, 1),
        "-AVERAGE_HR-": (20, 1), "-AMPLITUDE-": (20, 1),
    }
    for panel_layout in (researcher_hr_layout, processing_layout, signal_analysis_layout):
        for row in panel_layout:
            for element in row:
                element.Pad = (2, 2)
                if element.Key in value_sizes:
                    element.Size = value_sizes[element.Key]
                    element.AutoSizeText = False
                elif element.Type == sg.ELEM_TYPE_TEXT:
                    # Old STFT-caption case; unused by the current visible layout.
                    if element.DisplayText == "STFT: 4 s Hann, 0.5 s hop":
                        element.Size = (25, 2)
                        element.AutoSizeText = False
                    else:
                        element.Size = (None, None)
                        element.AutoSizeText = True

    sidebar_layout = [[sg.Frame(
        "", panel_layout, background_color=PANEL, border_width=1,
        expand_x=True, pad=(0, 4)
    )] for panel_layout in (researcher_hr_layout, signal_analysis_layout, processing_layout)]

    # Build one expandable panel for a Researcher plot canvas.
    def analysis_panel(key):
        return sg.Frame("", [[sg.Canvas(
            key=key, size=(420, 150), background_color=PANEL,
            expand_x=True, expand_y=True
        )]], background_color=PANEL, border_width=1, expand_x=True, expand_y=True)

    # ========================================================
    # RESEARCHER TAB
    # ========================================================

    researcher_layout = [
        [sg.Text("RESEARCHER / SIGNAL ANALYSIS", font=("Helvetica", 17, "bold"),
                 text_color=ACCENT, background_color=BG)],
        [sg.Text("PPG waveform and frequency analysis", text_color=TEXT_MUTED,
                 background_color=BG)],
        [sg.Frame("", comparison_plot_layout, background_color=PANEL,
                  border_width=1, expand_x=True, expand_y=True)],
        [analysis_panel("-FFT_CANVAS-")],
        [analysis_panel("-SPECTROGRAM_CANVAS-")],
        [analysis_panel("-DERIVATIVE_CANVAS-")],
        [sg.Column(sidebar_layout, key="-RESEARCH_SIDEBAR-", background_color=BG,
                   expand_x=True, expand_y=True, vertical_alignment="top")]
    ]

    # ========================================================
    # COMPLETE WINDOW
    # ========================================================

    layout = [
        header,
        [sg.HorizontalSeparator()],
        [
            sg.TabGroup(
                [[
                    sg.Tab(
                        "Monitor",
                        monitor_layout,
                        background_color=BG
                    ),
                    sg.Tab(
                        "Researcher",
                        researcher_layout,
                        background_color=BG
                    )
                ]],
                key="-MAIN_TABS-",
                expand_x=True,
                expand_y=True
            )
        ]
    ]

    window = sg.Window(
        "LumiPulse PPG Monitor",
        layout,
        background_color=BG,
        finalize=True,
        resizable=True
    )

    # POST-FINALISATION: configure the layout after FreeSimpleGUI creates the Tk widgets.
    configure_monitor_grid(window)
    configure_right_column(window)
    configure_researcher_grid(window)
    fit_initial_window(window)
    configure_host_hr_panel(window)
    configure_session_timer(window)
    configure_main_tabs(window)
    # The tab and font changes above can need a little more height than the first fit
    # allowed; grow the starting window (and its minimum) so no label is clipped.
    root = window.TKroot
    root.update_idletasks()
    needed = root.winfo_reqheight() + 12
    if needed > root.winfo_height():
        root.geometry(f"{root.winfo_width()}x{min(needed, root.maxsize()[1] - 32)}")
        root.minsize(root.minsize()[0], min(needed, root.maxsize()[1] - 32))
    # Let HOST text use only the height it needs; keep labels and wrapped values top-aligned.
    for row in host_processing_rows:
        for element in row:
            element.Widget.configure(height=0)  # HOST rows grow only when their text wraps.
            element.Widget.pack_configure(anchor="n")

    return window

# ============================================================
# DISPLAY HELPERS
# ============================================================

# Legends, finite-value scaling and Host HR status.

def style_legend(ax, handles, *, compact=False):
    """Style a PPG or HR-comparison legend using the LumiPulse colours.

    Author: Bianca Dibden (integration)
    """
    # Compact mode places the legend above the PPG plot.
    legend = ax.legend(
        handles=handles,
        loc="upper right" if compact else "lower right",
        bbox_to_anchor=(1.0, 1.15 if compact else 1.01),
        borderaxespad=0,
        borderpad=0.2 if compact else None,
        frameon=True,
        fontsize=10,
        ncol=2
    )
    legend.get_frame().set_facecolor(PANEL)
    legend.get_frame().set_edgecolor(BORDER)
    for legend_text in legend.get_texts():
        legend_text.set_color(TEXT)


def autoscale_y(ax, *arrays, minimum_span=20):
    """Fit the y-axis around finite data, adding a margin and a minimum visible span.

    Leave the previous limits unchanged when no finite data are available.
    Author: Bianca Dibden (integration)
    """
    values = np.concatenate([a[np.isfinite(a)] for a in arrays])
    if values.size == 0:
        return
    low, high = float(values.min()), float(values.max())
    # A minimum span prevents extreme zoom on flat data; add 10% headroom at each side.
    span = max(high - low, minimum_span)
    middle = (high + low) / 2
    ax.set_ylim(middle - 0.6 * span, middle + 0.6 * span)


def update_pulse_signal_box(window, signal_message, signal_ok):
    """Show the PPG pulse-signal check (from ppg_dsp.py) in its own box.

    This describes the sensor waveform only; Bluetooth link status is shown in the header.
    signal_message is 'Pulse signal good', 'Acquiring pulse signal' or three lines:
    'Check sensor', 'Reason: <all failed checks>' and the measured value against
    its threshold for each.
    """
    if signal_ok:
        state, colour, detail = "\u25cf GOOD", GREEN, "Clear pulse waveform"
    elif signal_message.startswith("Acquiring"):
        state, colour, detail = "\u25cf ACQUIRING", AMBER, "Settling - heart rate is not yet verified"
    else:
        state, colour = "\u25cf CHECK SENSOR", AMBER
        detail = signal_message.split("\n", 1)[-1]
    window["-PULSE_SIGNAL_STATE-"].update(state, text_color=colour)
    window["-PULSE_SIGNAL_DETAIL-"].update(detail, text_color=TEXT if not signal_ok else TEXT_MUTED)


def update_hr_display(window, host_hr, signal_message, low_limit, high_limit,
                      signal_ok=True):
    """Display Host HR to one decimal place and apply its visual alarm/status.

    Unavailable HR shows --.-. Available poor-quality HR stays visible; high/low alarms
    take priority over the amber quality warning. Return low/high/normal/invalid
    for alarm tracking; normal describes the HR range, not signal quality.
    Author: Bianca Dibden (integration)
    """
    update_pulse_signal_box(window, signal_message, signal_ok)
    if host_hr is None:
        window["-CURRENT_HR-"].update("--.-", text_color=TEXT_MUTED)
        window["-HR_STATUS-"].update("HEART RATE UNAVAILABLE", text_color=AMBER)
        return "invalid"

    window["-CURRENT_HR-"].update(f"{host_hr:.1f}")
    update_hr_status(window, host_hr, low_limit, high_limit)
    # High/low alarms take precedence; an in-range poor-quality estimate is shown amber below.
    if host_hr < low_limit:
        return "low"
    if host_hr > high_limit:
        return "high"
    if not signal_ok:
        # HR is still shown (testing), but flag that it is unverified; the reason is
        # in the PULSE SIGNAL box.
        window["-CURRENT_HR-"].update(text_color=AMBER)
        window["-HR_STATUS-"].update("HEART RATE UNVERIFIED", text_color=AMBER)
    return "normal"

# ============================================================
# MAIN PROGRAM
# ============================================================

# Run the GUI event loop: receive records, update displays and manage recording.
# Structure of main():
#   A. Setup      build window, create the four live plots and their data buffers
#   B. State      HR limits, recording variables, helper closures
#                 (set_recording, show_hr)
#   C. Start comms  create PPGLink and start its background thread
#   D. Loop (every ~50 ms):
#        1. read GUI events          2. log messages from comms.py
#        3. pull new packets (HostRecords) and buffer/record them
#        4. if there were new packets: redraw plots + update labels
#        5. update Researcher plots (max 1x/s), Bluetooth status, session timer
#        6. handle button/radio events (Apply Limits, Raw/Filtered/Both)
#   E. Shutdown   stop comms, close any open recording, wait for PDFs
def main():
    # Select a real serial port via argv, e.g. lp_gui.py COM5.
    # With no argument the built-in ESP32 simulator is used.
    port = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PORT

    window = create_gui()

    # ========================================================
    # CREATE PPG GRAPH
    # ========================================================

    ppg_fig, ppg_ax, ppg_fig_agg = add_plot(
        window["-PPG_CANVAS-"],
        figsize=(7.0, 2.25)
    )

    # The PPG plot is a fixed-length scrolling window: a 500-sample array whose x-axis
    # runs from -10 s (oldest) to 0 s (newest). Each packet shifts the array left
    # (np.roll) and writes the new samples at the right-hand end.
    # Keep newest samples on the right; the time slots assume the nominal 50 Hz sample rate.
    ppg_time = np.linspace(-PPG_DISPLAY_SECONDS, 0, PPG_BUFFER_SIZE)

    # NaN until real samples arrive, so no flat line is drawn at zero.
    raw_ppg_buffer = np.full(PPG_BUFFER_SIZE, np.nan)
    filtered_ppg_buffer = np.full(PPG_BUFFER_SIZE, np.nan)

    raw_ppg_line, = ppg_ax.plot(
        ppg_time, raw_ppg_buffer,
        label="Raw PPG", color=ACCENT, linewidth=1.5
    )
    filtered_ppg_line, = ppg_ax.plot(
        ppg_time, filtered_ppg_buffer,
        label="Filtered PPG", color=AMBER, linewidth=1.5
    )
    filtered_ppg_line.set_visible(False)

    ppg_ax.set_xlabel("Time (s)", labelpad=3)
    ppg_ax.set_ylabel("Amplitude (ADC counts)", labelpad=3)
    ppg_ax.set_xlim(-PPG_DISPLAY_SECONDS, 0)
    ppg_ax.set_ylim(0, 4095)
    ppg_ax.grid(True, alpha=0.2)
    style_legend(ppg_ax, [raw_ppg_line], compact=True)

    ppg_fig.subplots_adjust(left=0.11, right=0.98, bottom=0.21, top=0.90)
    ppg_fig_agg.draw()

    ppg_display_mode = "raw"

    # ========================================================
    # CREATE HR TREND GRAPH
    # ========================================================

    hr_fig, hr_ax, hr_fig_agg = add_plot(
        window["-HR_TREND_CANVAS-"],
        figsize=(7.0, 2.2)
    )

    # Three parallel lists, one entry per packet: GUI time, Host HR, MCU HR.
    # Entries older than 60 s are trimmed in the main loop; NaN draws as a gap.
    hr_times = []
    hr_values = []
    mcu_hr_values = []

    hr_line, = hr_ax.plot([], [], color=GREEN, linewidth=1.5)

    hr_ax.set_xlabel("Time (s)", labelpad=3)
    hr_ax.set_ylabel("Heart Rate (BPM)", labelpad=3)
    hr_ax.set_xlim(-HR_TREND_SECONDS, 0)
    hr_ax.set_ylim(40, 200)
    hr_ax.grid(True, alpha=0.2)
    hr_fig.subplots_adjust(left=0.11, right=0.98, bottom=0.22, top=0.96)
    hr_fig_agg.draw()

    # ========================================================
    # CREATE MCU vs HOST HR GRAPH (Researcher tab)
    # ========================================================

    cmp_fig, cmp_ax, cmp_fig_agg = add_plot(
        window["-HR_COMPARISON_CANVAS-"],
        figsize=(8.5, 2.6)
    )
    cmp_host_line, = cmp_ax.plot([], [], color=GREEN, linewidth=1.5, label="Host HR")
    cmp_mcu_line, = cmp_ax.plot([], [], color=ACCENT, linewidth=1.5,
                                linestyle="--", label="MCU HR")
    cmp_ax.set_xlabel("Time (s)", labelpad=3)
    cmp_ax.set_ylabel("Heart Rate (BPM)", labelpad=3)
    cmp_ax.set_xlim(-HR_TREND_SECONDS, 0)
    cmp_ax.set_ylim(40, 200)
    cmp_ax.grid(True, alpha=0.2)
    style_legend(cmp_ax, [cmp_host_line, cmp_mcu_line])
    cmp_fig.subplots_adjust(left=0.17, right=0.97, bottom=0.24, top=0.80)
    bind_researcher_mapping(cmp_fig_agg)
    researcher_draw_idle(cmp_fig_agg)

    researcher = ResearcherAnalysis()
    researcher_plots = create_researcher_plots(window)
    last_researcher_update = 0.0
    researcher_dirty = False
    reports = SessionReports()
    session_summary = None

    # ========================================================
    # PROGRAM VARIABLES
    # ========================================================

    # Measure GUI elapsed time with a clock unaffected by changes to the system date/time.
    program_start_time = time.monotonic()
    low_hr_limit = LOW_HR_LIMIT
    high_hr_limit = HIGH_HR_LIMIT
    current_hr = None
    latest_record = None
    hr_alarm_state = None

    # ========================================================
    # RECORDING VARIABLES
    # ========================================================
    # Recording is controlled exclusively by the ESP32 packet flag.
    recording_state = False
    recording_start_monotonic = None
    recording_start_time = None
    csv_file = None
    csv_writer = None
    csv_filename = None

    def set_recording(new_state):
        """Start or finalise a recording when the MCU recording flag changes.

        Normal shutdown can also call the same finalisation path.
        Author: Bianca Dibden (integration, based on Alva's recording code)
        """
        nonlocal recording_state, recording_start_monotonic, recording_start_time
        nonlocal csv_file, csv_writer, csv_filename, session_summary

        if new_state and not recording_state:
            recording_start_monotonic = time.monotonic()
            # CSV session time starts at the first saved sample (set below)
            recording_start_time = None
            csv_file, csv_writer, csv_filename = start_recording()
            session_summary = SessionSummary(csv_filename, low_hr_limit, high_hr_limit)
            window["-SESSION_RECORDING-"].update("● ON", text_color=RED)
            window["-SESSION_TIME-"].update("00:00:00")
            add_log(window, "Recording started.")

        elif not new_state and recording_state:
            # Close/flush CSV before submitting the independent PDF summary job.
            stop_recording(csv_file)
            window["-SESSION_RECORDING-"].update("● OFF", text_color=TEXT_MUTED)
            add_log(window, "Recording stopped.")
            if csv_filename is not None:
                add_log(window, "CSV saved: " + os.path.basename(csv_filename))
            csv_file = None
            csv_writer = None
            reports.submit(session_summary)
            session_summary = None

        recording_state = new_state

    def show_hr(host_hr, signal_message, signal_ok=True):
        """Update Host HR/status and log observed low/high alarm entries and recovery.

        Called for the latest record in a batch, or after applying new HR limits.
        Author: Bianca Dibden (integration)
        """
        nonlocal hr_alarm_state
        new_alarm_state = update_hr_display(
            window, host_hr, signal_message, low_hr_limit, high_hr_limit, signal_ok
        )
        if session_summary is not None:
            session_summary.alarm(new_alarm_state)
        # Log low/high entries and recovery once per observed transition, not per packet.
        if new_alarm_state != hr_alarm_state:
            if new_alarm_state == "low":
                add_log(window, "Pulse Low")
            elif new_alarm_state == "high":
                add_log(window, "Pulse High")
            elif new_alarm_state == "normal" and hr_alarm_state in ("low", "high"):
                add_log(window, "Pulse rate back to normal")
            hr_alarm_state = new_alarm_state

    # ========================================================
    # COMMUNICATION (comms.py)
    # ========================================================

    communication = {"shown": None}
    display_previous_record = None
    display_connection_count = 0
    display_interrupted = False

    add_log(window, "LumiPulse host application started.")
    if port == "SIM":
        add_log(window, "Simulation mode: replaying pulse.csv (no ESP32 needed).")
    else:
        add_log(window, f"Opening Bluetooth port {port}.")

    # PPGLink does all serial/Bluetooth I/O on its own thread. The GUI never touches the
    # port: it only asks the link for events/records and for its connection status.
    link = PPGLink(port)
    # Queue the initial limits; PPGLink sends and repeats them in its background thread.
    link.set_limits(low_hr_limit, high_hr_limit)   # ESP32 alarm uses the GUI limits
    link.start()

    # ========================================================
    # MAIN EVENT LOOP
    # ========================================================

    while True:

        # Process events and queued records; 50 ms polling does not guarantee 1 Hz plotting.
        event, values = window.read(timeout=GUI_POLL_MS)

        if event == sg.WIN_CLOSED:
            break

        now = time.monotonic()
        elapsed_time = now - program_start_time
        # Remember interruptions even when this poll receives no new records.
        connections = link.stats["connects"]
        if (link.status != STATUS_CONNECTED or link.no_data_alarm
                or connections != display_connection_count):
            display_interrupted = True
        display_connection_count = connections

        # ----------------------------------------------------
        # LOG EVENTS FROM COMMS (connect, lost, invalid, ...)
        # ----------------------------------------------------
        for line in link.get_events():
            # Recording start/stop is logged by the GUI itself (with the CSV
            # file name), so skip the duplicate lines from comms.py.
            if line.endswith(("Recording started", "Recording stopped")):
                continue
            add_log_line(window, line)

        # ----------------------------------------------------
        # NEW PACKETS (normally one per second)
        # ----------------------------------------------------
        # Record and buffer every queued record; render the latest state once per batch.
        records = link.get_records()

        for record in records:
            # Per-packet pipeline: (1) note packet arrival, (2) start/stop/save the CSV,
            # (3) detect data gaps, (4) shift the PPG buffers, (5) append to HR history.
            register_valid_packet(window, record)

            # Recording state from the ESP32 button
            first_sample_time = record.time - (len(record.raw) - 1) * SAMPLE_PERIOD
            set_recording(record.recording)
            if recording_state and csv_writer is not None:
                if recording_start_time is None:
                    recording_start_time = first_sample_time
                save_packet(csv_writer, record, recording_start_time)
                session_summary.observe(record)

            # Mark missing data only in plot history; never add these NaNs to DSP or recording.
            missing = display_packet_gap(display_previous_record, record,
                                         display_interrupted, link.alarm_after)
            interrupted = missing is None or missing > 0
            if interrupted:
                hr_times.append(elapsed_time)
                hr_values.append(np.nan)
                mcu_hr_values.append(np.nan)
                if missing is None:
                    raw_ppg_buffer.fill(np.nan)
                    filtered_ppg_buffer.fill(np.nan)
                else:
                    # Slots mark absent measurements in the display only.
                    slots = min(missing, PPG_BUFFER_SIZE)
                    raw_ppg_buffer = np.roll(raw_ppg_buffer, -slots)
                    filtered_ppg_buffer = np.roll(filtered_ppg_buffer, -slots)
                    raw_ppg_buffer[-slots:] = np.nan
                    filtered_ppg_buffer[-slots:] = np.nan
            researcher.append(record, interrupted=interrupted)
            researcher_dirty = True
            display_previous_record = record

            # Drop old samples and append this packet (normally 50 samples).
            n = len(record.raw)
            raw_ppg_buffer = np.roll(raw_ppg_buffer, -n)
            raw_ppg_buffer[-n:] = record.raw
            filtered_ppg_buffer = np.roll(filtered_ppg_buffer, -n)
            filtered_ppg_buffer[-n:] = record.filtered

            # HR history (NaN = no valid value, shown as a gap)
            current_hr = record.host_hr
            hr_times.append(elapsed_time)
            hr_values.append(np.nan if record.host_hr is None else record.host_hr)
            mcu_hr_values.append(np.nan if record.mcu_hr is None else record.mcu_hr)

            latest_record = record

        # Everything below runs only when new packets arrived, i.e. about once per second.
        # Only the newest record's values are shown on the labels.
        if records:
            # A reconnect cannot be placed precisely within a queued batch.
            # Separate its records to avoid joining uncertain data, then resume normal history.
            display_interrupted = False
            rec = latest_record

            # ------------------------------------------------
            # REDRAW PPG GRAPH (Reqts 7, 8)
            # ------------------------------------------------
            if ppg_display_mode == "raw":
                raw_ppg_line.set_ydata(raw_ppg_buffer)
                autoscale_y(ppg_ax, raw_ppg_buffer)
            elif ppg_display_mode == "filtered":
                filtered_ppg_line.set_ydata(filtered_ppg_buffer)
                autoscale_y(ppg_ax, filtered_ppg_buffer)
            else:
                # Remove the displayed raw mean for comparison; recorded raw data stay unchanged.
                raw_centred = raw_ppg_buffer - (
                    np.nanmean(raw_ppg_buffer) if np.isfinite(raw_ppg_buffer).any() else 0
                )
                raw_ppg_line.set_ydata(raw_centred)
                filtered_ppg_line.set_ydata(filtered_ppg_buffer)
                autoscale_y(ppg_ax, raw_centred, filtered_ppg_buffer)
            ppg_fig_agg.draw()

            # ------------------------------------------------
            # CURRENT HR + ALARM (Reqts 9, 10, 11)
            # ------------------------------------------------
            show_hr(rec.host_hr, rec.signal_message, rec.signal_ok)

            # ------------------------------------------------
            # KEEP ONLY THE MOST RECENT 60 s OF HR
            # ------------------------------------------------
            while hr_times and elapsed_time - hr_times[0] > HR_TREND_SECONDS:
                hr_times.pop(0)
                hr_values.pop(0)
                mcu_hr_values.pop(0)

            # Put HR estimates on a time-before-now axis; queued records can share a GUI timestamp.
            display_hr_times = np.array(hr_times) - elapsed_time

            hr_line.set_data(display_hr_times, hr_values)
            hr_fig_agg.draw()

            cmp_host_line.set_data(display_hr_times, hr_values)
            cmp_mcu_line.set_data(display_hr_times, mcu_hr_values)
            researcher_draw_idle(cmp_fig_agg)

            # ------------------------------------------------
            # RESEARCHER TAB
            # ------------------------------------------------
            fmt_bpm = lambda v: "--.- BPM" if v is None else f"{v:.1f} BPM"
            window["-MCU_HR-"].update(fmt_bpm(rec.mcu_hr))
            window["-HOST_HR-"].update(fmt_bpm(rec.host_hr))
            if rec.mcu_hr is not None and rec.host_hr is not None:
                window["-HR_DIFFERENCE-"].update(f"{rec.host_hr - rec.mcu_hr:+.1f} BPM")
            else:
                window["-HR_DIFFERENCE-"].update("--.- BPM")
            window["-FILTER_STATUS-"].update(
                f"Band-pass 0.5-5 Hz, {rec.latency_s * 1000:.1f} ms/packet"
            )
            window["-SIGNAL_QUALITY-"].update(
                rec.signal_message, text_color=GREEN if rec.signal_ok else AMBER
            )
            window["-PULSE_COUNT-"].update("--")
            amplitude = researcher.amplitude()
            window["-AMPLITUDE-"].update("--" if amplitude is None else f"{amplitude:.1f} ADC counts")

            # Average available Host HR in the rolling history, not the whole recording session.
            valid_hr = [v for v in hr_values if np.isfinite(v)]
            window["-AVERAGE_HR-"].update(
                f"{np.mean(valid_hr):.1f} BPM" if valid_hr else "--"
            )

        # Refresh dirty Researcher analysis no more often than once per second.
        # This does not guarantee a minimum update rate; Monitor updates are packet-driven.
        if researcher_dirty and now - last_researcher_update >= 1.0:
            update_researcher_plots(researcher_plots, researcher)
            last_researcher_update = now
            researcher_dirty = False
        reports.poll(window)

        # ----------------------------------------------------
        # BLUETOOTH STATUS + NO-DATA ALARM (Reqt 14)
        # ----------------------------------------------------
        update_communication_display(window, communication, link)
        # Suppress old amplitude during the no-data alarm, including after queued-record updates.
        if link.no_data_alarm:
            window["-AMPLITUDE-"].update("--")

        # ----------------------------------------------------
        # SESSION TIMER
        # ----------------------------------------------------
        if recording_state and recording_start_monotonic is not None:
            session_elapsed = now - recording_start_monotonic
            hours = int(session_elapsed // 3600)
            minutes = int((session_elapsed % 3600) // 60)
            seconds = int(session_elapsed % 60)
            window["-SESSION_TIME-"].update(f"{hours:02d}:{minutes:02d}:{seconds:02d}")

        # ----------------------------------------------------
        # APPLY USER-DEFINED HR LIMITS
        # ----------------------------------------------------
        if event == "-APPLY_LIMITS-":
            try:
                new_low_limit = float(values["-LOW_HR_LIMIT-"])
                new_high_limit = float(values["-HIGH_HR_LIMIT-"])
                # Reject non-finite limits before changing settings; clear the warning only on valid apply.
                if not (np.isfinite(new_low_limit) and np.isfinite(new_high_limit)):
                    window["-HR_LIMIT_ERROR-"].update("Invalid HR limits")
                    add_log(window, "Invalid HR limits: enter finite numeric values.")
                elif new_low_limit >= new_high_limit:
                    window["-HR_LIMIT_ERROR-"].update("Invalid HR limits")
                    add_log(window, "Invalid HR limits: low must be below high.")
                else:
                    window["-HR_LIMIT_ERROR-"].update("")
                    low_hr_limit = new_low_limit
                    high_hr_limit = new_high_limit
                    if session_summary is not None:
                        session_summary.change_limits(low_hr_limit, high_hr_limit)
                    # The device message has its own range/rounding rules; a rejection is logged.
                    try:
                        link.set_limits(low_hr_limit, high_hr_limit)
                    except ValueError:
                        add_log(window, "HR limits not sent to the monitor (outside 0-999.9 BPM).")
                    add_log(
                        window,
                        f"HR limits updated: {low_hr_limit:.1f}-{high_hr_limit:.1f} BPM."
                    )
                    if latest_record is not None and not link.no_data_alarm:
                        show_hr(current_hr, latest_record.signal_message,
                                latest_record.signal_ok)
            except ValueError:
                window["-HR_LIMIT_ERROR-"].update("Invalid HR limits")
                add_log(window, "Invalid HR limits: enter numeric values.")

        # ----------------------------------------------------
        # PPG DISPLAY SELECTION
        # ----------------------------------------------------
        if event in ("-SHOW_RAW-", "-SHOW_FILTERED-", "-SHOW_BOTH-"):
            ppg_display_mode = {
                "-SHOW_RAW-": "raw",
                "-SHOW_FILTERED-": "filtered",
                "-SHOW_BOTH-": "both",
            }[event]

            raw_ppg_line.set_visible(ppg_display_mode in ("raw", "both"))
            filtered_ppg_line.set_visible(ppg_display_mode in ("filtered", "both"))

            if ppg_display_mode == "raw":
                raw_ppg_line.set_label("Raw PPG")
                raw_ppg_line.set_ydata(raw_ppg_buffer)
                ppg_ax.set_ylabel("Amplitude (ADC counts)", labelpad=3)
                style_legend(ppg_ax, [raw_ppg_line], compact=True)
                autoscale_y(ppg_ax, raw_ppg_buffer)
            elif ppg_display_mode == "filtered":
                filtered_ppg_line.set_ydata(filtered_ppg_buffer)
                ppg_ax.set_ylabel("Amplitude (ADC counts)", labelpad=3)
                style_legend(ppg_ax, [filtered_ppg_line], compact=True)
                autoscale_y(ppg_ax, filtered_ppg_buffer)
            else:
                raw_centred = raw_ppg_buffer - (
                    np.nanmean(raw_ppg_buffer) if np.isfinite(raw_ppg_buffer).any() else 0
                )
                raw_ppg_line.set_label("Raw PPG (DC removed)")
                raw_ppg_line.set_ydata(raw_centred)
                filtered_ppg_line.set_ydata(filtered_ppg_buffer)
                ppg_ax.set_ylabel("Amplitude (ADC counts)", labelpad=3)
                style_legend(ppg_ax, [raw_ppg_line, filtered_ppg_line], compact=True)
                autoscale_y(ppg_ax, raw_centred, filtered_ppg_buffer)

            ppg_fig_agg.draw()
            add_log(window, f"PPG display changed to {ppg_display_mode.capitalize()}.")

    # ========================================================
    # NORMAL SHUTDOWN AND SESSION FINALISATION
    # ========================================================

    # Stop reception and finalise the session; queued records are not drained on this path.
    link.stop()
    if csv_file is not None:
        # Closing while recording uses the same transition/finalization path.
        set_recording(False)
    reports.close(window)
    window.close()

# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()
