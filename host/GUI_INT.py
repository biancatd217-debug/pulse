"""
LumiPulse PPG Monitor
BMET2922 Major Project

Role 3:
- FreeSimpleGUI interface
- Rolling HR and PPG plots
- User-defined HR limits and alarms
- System log
- Recording-state handling
- CSV recording
- Researcher tab
- Communication-loss alarm

INTEGRATED VERSION:
Live data comes from comms.py (PPGLink): Bluetooth packets are received,
validated and processed by ppg_dsp.py, and each second the GUI receives
communication status, MCU HR, recording state, raw + filtered PPG, host HR
and signal quality, plus timestamped log events.

GUI_INT.py is the integrated version of Alva's gui.py (layout unchanged;
synthetic data replaced by live data from comms.py / ppg_dsp.py).

Run (from Code/host):
    py GUI_INT.py            simulated ESP32 (replays pulse.csv) - no hardware
    py GUI_INT.py COM5       real ESP32 on Bluetooth COM5 (Windows)
    python3 GUI_INT.py /dev/tty.BMET2922-G00   (Mac)

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

FONT_TITLE = ("Helvetica", 22, "bold")
FONT_SECTION = ("Helvetica", 12, "bold")
FONT_NORMAL = ("Helvetica", 11)
FONT_SMALL = ("Helvetica", 11, "bold")
FONT_HR = ("Helvetica", 80, "bold")
FONT_VALUE = ("Helvetica", 15, "bold")

# ============================================================
# SYSTEM SETTINGS
# ============================================================

# ESP32 samples every 20 ms = 50 Hz.
SAMPLE_RATE = 50
SAMPLE_PERIOD = 1 / SAMPLE_RATE

# PPG shows the most recent 10 seconds.

PPG_DISPLAY_SECONDS = 10
PPG_BUFFER_SIZE = SAMPLE_RATE * PPG_DISPLAY_SECONDS

# Redraw PPG at 10 Hz.
PPG_PLOT_RATE = 10
PPG_PLOT_PERIOD = 1 / PPG_PLOT_RATE

# HR display updates once per second.
HR_UPDATE_PERIOD = 1.0

# HR trend shows most recent 60 seconds.
HR_TREND_SECONDS = 60

# Default HR limits. User can override these.
LOW_HR_LIMIT = 60
HIGH_HR_LIMIT = 100

# Communication alarm (5 s, Reqt 14) is timed in comms.py (ALARM_AFTER_S).

# Port used when none is given on the command line ("SIM" = simulator).
DEFAULT_PORT = "SIM"

# How often the event loop checks for new data (ms).
GUI_POLL_MS = 50

# ============================================================
# MATPLOTLIB HELPER
# ============================================================

def add_plot(canvas_element, figsize):
    """
    Creates and embeds a Matplotlib plot inside a GUI Canvas.
    """

    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor(PANEL)
    ax.set_facecolor(PANEL)
    ax.tick_params(
        colors=TEXT_MUTED,
        labelsize=9
    )

    ax.xaxis.label.set_color(TEXT_MUTED)
    ax.yaxis.label.set_color(TEXT_MUTED)

    for spine in ax.spines.values():
        spine.set_color(BORDER)

    # The GUI allocates this viewport. Do not let the embedded TkAgg
    # widget feed its requested size back into the surrounding grid.
    canvas_element.TKCanvas.pack_propagate(False)

    fig_agg = FigureCanvasTkAgg(
        fig,
        canvas_element.TKCanvas
    )

    fig_agg.draw()
    # TkAgg owns Configure handling and coalesces resize redraws via draw_idle.
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
    """One-sided Hann-window FFT amplitude, in filtered ADC counts."""
    x = np.asarray(samples, dtype=float)
    if len(x) < 2 * SAMPLE_RATE or not np.isfinite(x).all():
        return np.array([]), np.array([])
    window = np.hanning(len(x))
    magnitude = 2 * np.abs(np.fft.rfft((x - x.mean()) * window)) / window.sum()
    magnitude[0] *= .5
    if len(x) % 2 == 0:
        magnitude[-1] *= .5
    frequency = np.fft.rfftfreq(len(x), SAMPLE_PERIOD)
    keep = frequency <= 5
    return frequency[keep], magnitude[keep]


def ppg_spectrogram(samples):
    """4 s Hann frames (200 samples), 0.5 s hop (25), 87.5% overlap.

    Uses the existing filtered PPG, with per-frame DC removal. Display dB
    magnitude relative to 1 ADC count; no physiological interpretation.
    Frame-centre times are seconds before the end of the rolling window.
    """
    x = np.asarray(samples, dtype=float)
    length, hop = 200, 25
    if len(x) < length or not np.isfinite(x).all():
        return None
    frames = np.lib.stride_tricks.sliding_window_view(x, length)[::hop]
    window = np.hanning(length)
    magnitude = 2 * np.abs(np.fft.rfft(
        (frames - frames.mean(axis=1, keepdims=True)) * window, axis=1
    )) / window.sum()
    magnitude[:, 0] *= .5
    magnitude[:, -1] *= .5
    frequency = np.fft.rfftfreq(length, SAMPLE_PERIOD)
    keep = frequency <= 5
    centres = (np.arange(len(frames)) * hop + (length - 1) / 2 - (len(x) - 1)) * SAMPLE_PERIOD
    return frequency[keep], centres, 20 * np.log10(np.maximum(magnitude[:, keep].T, 1e-6))


def display_packet_gap(previous, record, interrupted=False, timeout=5.0):
    """Missing display slots, or None for an unknown-duration interruption.

    Reception-time gaps are conservative: they do not measure acquisition time.
    Never modify the received record or infer samples from an uncertain restart.
    """
    if interrupted or len(record.raw) != SAMPLES_PER_PACKET or len(record.filtered) != SAMPLES_PER_PACKET:
        return None
    if previous is None:
        return 0
    if len(previous.raw) != SAMPLES_PER_PACKET or len(previous.filtered) != SAMPLES_PER_PACKET:
        return None
    step = (record.seq - previous.seq) % SEQ_MODULO
    if 1 < step <= MAX_FORWARD_GAP:
        return (step - 1) * SAMPLES_PER_PACKET
    if step != 1 or record.time - previous.time >= timeout:
        return None
    return 0


class ResearcherAnalysis:
    """Ten seconds of continuous received filtered data; reset across gaps."""
    def __init__(self):
        self.samples = deque(maxlen=10 * SAMPLE_RATE)
        self.last_seq = None
        self.valid_samples = 0

    def append(self, record, interrupted=False):
        if interrupted or (self.last_seq is not None and (record.seq - self.last_seq) % 65536 != 1):
            self.samples.clear()
            self.valid_samples = 0
        self.last_seq = record.seq
        self.samples.extend(record.filtered)
        self.valid_samples = self.valid_samples + len(record.filtered) if record.signal_ok else 0

    def amplitude(self):
        # Recent relative amplitude = peak-to-peak of the last 3 s of filtered
        # ADC counts; require three seconds of consecutive quality-approved data.
        if self.valid_samples < 3 * SAMPLE_RATE:
            return None
        return float(np.ptp(np.asarray(self.samples)[-3 * SAMPLE_RATE:]))


def researcher_draw_idle(canvas):
    """Queue visible Researcher draws; hidden artists still receive all data."""
    if canvas.get_tk_widget().winfo_ismapped():
        canvas.draw_idle()


def bind_researcher_mapping(canvas):
    """Keep native TkAgg resizing, ignoring redundant same-size notifications.

    Tk can deliver repeated Configure events for a mapped tab without a size
    change. TkAgg otherwise deletes/recreates its backing image every time,
    producing blank frames. Only genuine size/DPI changes need that work.
    """
    widget = canvas.get_tk_widget()
    last_size = None

    def resize_if_changed(event):
        nonlocal last_size
        size = (event.width, event.height, canvas.device_pixel_ratio)
        if size == last_size:
            return
        last_size = size
        canvas.resize(event)

    # Replace only TkAgg's Configure binding, retaining its own resize method.
    # No dimensions are assigned and no update()/nested event loop is involved.
    widget.bind('<Configure>', resize_if_changed)
    widget.bind('<Map>', lambda event: canvas.draw_idle(), add='+')


def create_researcher_plots(window):
    """Create each Figure, TkAgg canvas and artist exactly once."""
    result = {}
    specs = (
        ('fft', '-FFT_CANVAS-', 'Frequency Spectrum (Last 10 s)', 'Frequency (Hz)', 'Spectral Amplitude (ADC counts)'),
        ('spectrogram', '-SPECTROGRAM_CANVAS-', 'Spectrogram — Last 10 s', 'Time (s)', 'Frequency (Hz)'),
        ('derivative', '-DERIVATIVE_CANVAS-', 'Second Derivative (PPG Acceleration)', 'Time (s)', 'ADC counts/s^2'),
    )
    for name, key, title, xlabel, ylabel in specs:
        fig, ax, canvas = add_plot(window[key], (4.5, 1.7))
        ax.set_title(title, color=TEXT, fontsize=11)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        if name == 'derivative':
            # Move only the label slightly left; retain axes/figure bounds.
            ax.yaxis.labelpad = 0
        fig.subplots_adjust(left=.17, right=.97, bottom=.24, top=.85)
        if name == 'spectrogram':
            # Fixed display cells for the unchanged 200-sample / 25-hop STFT.
            # Right-align partial histories instead of resizing the image/extent.
            # Use the existing STFT centres + 10 s: 2.01..8.01 s at full
            # history. Half-hop cell edges are 1.76..8.26 s. Show that valid
            # centre-time domain without stretching estimates to 0..10 s or
            # inventing boundary columns. Partial histories remain NaN-padded.
            ax.set_position([.16, .24, .60, .61])
            artist = ax.imshow(np.full((21, 13), np.nan), origin='lower', aspect='auto',
                               extent=(1.76, 8.26, -.125, 5.125),
                               vmin=-40, vmax=60, cmap='cividis', interpolation='nearest')
            artist.set_visible(False)
            # Explicit colourbar axes avoid automatic parent-axes repositioning.
            colourbar_axes = fig.add_axes([.80, .24, .025, .61])
            bar = fig.colorbar(artist, cax=colourbar_axes)
            bar.set_label('Magnitude (dB re 1 ADC count)', color=TEXT_MUTED, fontsize=7)
            bar.ax.tick_params(colors=TEXT_MUTED, labelsize=7)
            ax.set_xlim(1.76, 8.26)
            ax.set_ylim(0, 5)
        else:
            artist, = ax.plot([], [], color=ACCENT, linewidth=1)
            ax.set_xlim((0, 5) if name == 'fft' else (0, 10))
            ax.grid(True, alpha=.2)
        waiting = ax.text(.5, .5, 'Acquiring continuous data', transform=ax.transAxes,
                          ha='center', color=TEXT_MUTED, fontsize=9)
        bind_researcher_mapping(canvas)
        result[name] = (ax, artist, canvas, waiting)
    return result


def update_researcher_plots(plots, analysis):
    """One update per second at most; no widget/figure recreation."""
    x = np.asarray(analysis.samples, dtype=float)
    ax, line, canvas, waiting = plots['fft']
    freq, magnitude = spectral_magnitude(x)
    line.set_data(freq, magnitude)
    waiting.set_visible(len(freq) == 0)
    if len(freq):
        ax.set_ylim(0, max(1, float(magnitude.max()) * 1.15))
    researcher_draw_idle(canvas)

    ax, image, canvas, waiting = plots['spectrogram']
    spec = ppg_spectrogram(x)
    image.set_visible(spec is not None)
    waiting.set_visible(spec is None)
    if spec is not None:
        freq, centres, db = spec
        display = np.full((21, 13), np.nan)
        display[:, -db.shape[1]:] = db
        image.set_data(display)
    researcher_draw_idle(canvas)

    ax, line, canvas, waiting = plots['derivative']
    if len(x) >= SAMPLE_RATE:
        # Two centred numerical gradients with dt=0.02 s. Hide the two edge
        # samples at either end, where one-sided differences are less reliable.
        acceleration = np.gradient(np.gradient(x, SAMPLE_PERIOD, edge_order=2),
                                   SAMPLE_PERIOD, edge_order=2)
        acceleration[:2] = acceleration[-2:] = np.nan
        # Display coordinates only: newest sample at 10 s, older samples
        # step left by 20 ms; the fixed axis never moves.
        line.set_data(10 + (np.arange(len(x)) - len(x) + 1) * SAMPLE_PERIOD,
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
    """Metadata from actual saved packets and actual GUI alarm decisions."""
    def __init__(self, filename, low, high):
        self.filename = filename
        self.samples = 0
        self.quality = Counter()
        self.limits = [(0.0, low, high)]
        self.alarm_events = Counter()
        self.last_alarm = None
        self.start_time = None
        self.last_time = None
        self.packet_count = 0
        self.good_packets = 0
        self.valid_samples = 0
        self.gaps = 0
        self.last_seq = None

    def observe(self, record):
        if self.start_time is None:
            self.start_time = record.time - (len(record.raw) - 1) * SAMPLE_PERIOD
        self.last_time = record.time
        self.samples += len(record.raw)
        self.packet_count += 1
        self.good_packets += int(record.signal_ok)
        self.valid_samples = self.valid_samples + len(record.filtered) if record.signal_ok else 0
        self.quality[record.signal_message] += 1
        if self.last_seq is not None and (record.seq - self.last_seq) % 65536 != 1:
            self.gaps += 1
            self.valid_samples = len(record.filtered) if record.signal_ok else 0
        self.last_seq = record.seq

    def alarm(self, state):
        if state in ('low', 'high') and state != self.last_alarm:
            self.alarm_events[state] += 1
        self.last_alarm = state

    def change_limits(self, low, high):
        elapsed = max(0, time.time() - self.start_time) if self.start_time is not None else 0
        self.limits.append((elapsed, low, high))


def generate_session_pdf(session):
    """Read the CLOSED CSV, render independent report Figures with PdfPages.

    Runs in one background worker: no Tk calls, shared GUI artists, pyplot,
    backend changes, or reprocessing of Host HR.
    """
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
            fig = Figure(figsize=(8.27, 11.69), facecolor='white')
            fig.text(.09, .94, 'LumiPulse PPG Monitor', fontsize=23, weight='bold', color='#253c51')
            fig.text(.09, .905, 'Session Report', fontsize=16)
            fig.text(.09, .875, rows[0]['Timestamp'], fontsize=10, color='#555555')
            quality_text = '; '.join(f'{message}: {count}' for message, count in session.quality.items())
            quality_lines = textwrap.wrap(quality_text, width=88)
            lines = [
                'SESSION',
                f'Duration (receipt-time estimate): {duration:.2f} s',
                f'Recorded samples: {len(rows)}   |   Sampling rate: {SAMPLE_RATE} Hz',
                f'Recorded signal coverage: {len(rows) / SAMPLE_RATE:.2f} s   |   Discontinuities: {session.gaps}',
                f'Initial HR limits: {session.limits[0][1]:g} - {session.limits[0][2]:g} BPM',
                f'Limit changes during recording: {len(session.limits) - 1}',
                f'Final HR limits: {session.limits[-1][1]:g} - {session.limits[-1][2]:g} BPM',
                '', 'HEART RATE (valid per-packet estimates)',
                f'Host average / minimum / maximum: {fmt(mean(valid_host))} / '
                f'{fmt(float(valid_host.min()) if len(valid_host) else None)} / '
                f'{fmt(float(valid_host.max()) if len(valid_host) else None)} BPM',
                f'MCU average: {fmt(mean(valid_mcu))} BPM',
                f'Mean Host - MCU difference (paired): {fmt(mean(difference))} BPM',
                f'Valid Host HR packets: {len(valid_host)} / {session.packet_count}',
                '', 'SIGNAL / QUALITY',
                f'Quality-approved packets: {session.good_packets} / {session.packet_count}',
                *quality_lines[:3],
                *( ['Full quality/limit history follows the signal views.'] if len(quality_lines) > 3 else []),
                '', 'OBSERVED GUI HR ALARM ENTRIES',
                f'Low: {session.alarm_events["low"]}   |   High: {session.alarm_events["high"]}',
                'Counts reflect received valid HR and the limits active at that time.',
                '', 'NOTES',
                'Sample timestamps are reconstructed from host receipt time.',
                'PPG amplitude is relative ADC counts, not calibrated optical amplitude.',
                'Frequency plots are descriptive; no respiratory metric is inferred.',
            ]
            y = .83
            for line in lines:
                fig.text(.09, y, line, fontsize=10 if line else 5,
                         weight='bold' if line.isupper() else 'normal')
                y -= .024 if line else .013
            fig.text(.09, .045, os.path.basename(session.filename) + '  |  Page 1', fontsize=8, color='#666666')
            pdf.savefig(fig)

            fig = Figure(figsize=(8.27, 11.69), facecolor='white')
            fig.text(.09, .955, 'Recorded session - signal views', fontsize=18, weight='bold', color='#253c51')
            axes = fig.subplots(3, 1)
            fig.subplots_adjust(left=.12, right=.94, top=.91, bottom=.10, hspace=.48)
            # Select the final continuous segment using sequence and timestamp
            # discontinuities. Never join samples from opposite sides of a gap.
            seq = np.array([int(row['Packet Seq']) for row in rows])
            boundaries = np.flatnonzero((np.diff(times) > .1) | (np.diff(times) < 0) |
                ((np.diff(seq) != 0) & ((np.diff(seq) % 65536) != 1))) + 1
            segment_start = int(boundaries[-1]) if len(boundaries) else 0
            start = max(segment_start, len(rows) - 10 * SAMPLE_RATE)
            view_time = times[start:]
            axes[0].plot(view_time, raw[start:] - np.mean(raw[start:]), label='Raw (DC removed)', lw=.8)
            axes[0].plot(view_time, filtered[start:], label='Filtered (existing pipeline)', lw=.8)
            axes[0].set_title('PPG - final continuous portion, up to 10 s', fontsize=11)
            axes[0].set_ylabel('ADC counts')
            axes[0].set_xlabel('Session time (s)')
            axes[0].legend(fontsize=8)
            # Break only at sequence discontinuities; receipt jitter is not loss.
            hr_breaks = np.flatnonzero(np.diff(seq[packet_end]) % SEQ_MODULO != 1) + 1
            hr_time = times[packet_end]
            plot_time = np.insert(hr_time, hr_breaks, hr_time[hr_breaks])
            plot_host = np.insert(host[packet_end], hr_breaks, np.nan)
            plot_mcu = np.insert(mcu[packet_end], hr_breaks, np.nan)
            axes[1].plot(plot_time, plot_host, label='Host HR', lw=1)
            axes[1].plot(plot_time, plot_mcu, '--', label='MCU HR', lw=1)
            axes[1].set_title('HR trend and MCU / Host comparison', fontsize=11)
            axes[1].set_ylabel('BPM')
            axes[1].set_xlabel('Session time (s)')
            axes[1].legend(fontsize=8)
            frequency, magnitude = spectral_magnitude(filtered[start:])
            axes[2].plot(frequency, magnitude, lw=1)
            axes[2].set_xlim(0, 5)
            axes[2].set_title('Spectrum - recorded final continuous portion', fontsize=11)
            axes[2].set_xlabel('Frequency (Hz)')
            axes[2].set_ylabel('Spectral magnitude (ADC)')
            if not len(frequency):
                axes[2].text(.5, .5, 'Insufficient continuous data (need 2 s)', ha='center', transform=axes[2].transAxes)
            amplitude = (float(np.ptp(filtered[-150:]))
                         if len(rows) - segment_start >= 150 and session.valid_samples >= 150 else None)
            fig.text(.09, .05, f'Recent filtered peak-to-peak (3 s, quality-approved): {fmt(amplitude)} ADC counts', fontsize=9)
            fig.text(.09, .03, os.path.basename(pdf_path) + '  |  Page 2', fontsize=8, color='#666666')
            for ax in axes:
                ax.grid(True, alpha=.2)
                ax.tick_params(labelsize=8)
            pdf.savefig(fig)
            if len(session.limits) > 1 or len(quality_lines) > 3:
                details = ['HR limit history (session seconds / low / high BPM)']
                details += [f'{elapsed:.2f} s   {low:g} - {high:g}'
                            for elapsed, low, high in session.limits]
                details += ['', 'Signal-quality packet counts'] + quality_lines
                for offset in range(0, len(details), 38):
                    fig = Figure(figsize=(8.27, 11.69), facecolor='white')
                    fig.text(.09, .94, 'Session metadata', fontsize=18, weight='bold')
                    for index, line in enumerate(details[offset:offset + 38]):
                        fig.text(.09, .88 - index * .021, line, fontsize=10)
                    fig.text(.09, .04, f'Page {3 + offset // 38}', fontsize=8)
                    pdf.savefig(fig)
        os.replace(temporary_path, pdf_path)
    except Exception:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise
    return pdf_path


class SessionReports:
    """Submit each closed session once and report results on the GUI thread."""
    def __init__(self):
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='LumiPulsePDF')
        self.pending = []
        self.submitted = set()

    def submit(self, session):
        if session is None or not session.samples or session.filename in self.submitted:
            return
        self.submitted.add(session.filename)
        self.pending.append(self.worker.submit(generate_session_pdf, session))

    def poll(self, window):
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
        self.worker.shutdown(wait=True)
        self.poll(window)

# ============================================================
# SYSTEM LOG
# ============================================================

def add_log(window, message):
    """
    Adds an event to the system log (Reqt 13).
    Required format example:
    Thu Sep 19 17:46:50 2024: Pulse Low
    Uses time.ctime() so GUI events match the comms.py events exactly.
    """
    add_log_line(window, f"{time.ctime()}: {message}")


def add_log_line(window, line):
    """
    Appends a line that is already timestamped, e.g. from
    PPGLink.get_events() in comms.py.

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
    """
    Call this for every new valid packet (HostRecord from PPGLink).
    Updates the 'Last packet' field on the Researcher tab.
    """
    packet_time = datetime.fromtimestamp(record.time).strftime("%H:%M:%S")
    window["-LAST_PACKET-"].update(f"{packet_time}  (#{record.seq})")


def update_communication_display(window, communication, link):
    """
    Shows the Bluetooth status and the no-data alarm (Reqt 14) from
    PPGLink. Only redraws when something changes.

    Author: Bianca Dibden (integration)
    """
    # Never leave a stale HR on screen while there is no data.
    # The next valid packet overwrites this.
    if link.no_data_alarm:
        window["-CURRENT_HR-"].update("--.-", text_color=TEXT_MUTED)
        window["-HR_STATUS-"].update("NO DATA - CHECK CONNECTION", text_color=RED)

    state = (link.status, link.no_data_alarm)
    if state == communication["shown"]:
        return
    communication["shown"] = state

    if link.status == STATUS_CONNECTED and not link.no_data_alarm:
        window["-BT_STATUS-"].update("● CONNECTED", text_color=GREEN)
    elif link.status == STATUS_CONNECTED:
        window["-BT_STATUS-"].update("● CONNECTED - NO DATA", text_color=AMBER)
    elif link.status == STATUS_CONNECTING:
        window["-BT_STATUS-"].update("● CONNECTING...", text_color=AMBER)
    else:
        window["-BT_STATUS-"].update("● DISCONNECTED", text_color=RED)

    window["-COMM_ALARM-"].update(
        "⚠ NO DATA FOR 5 s" if link.no_data_alarm else ""
    )

# ============================================================
# CSV RECORDING
# ============================================================

def start_recording():
    """
    Creates a new CSV file when recording starts.
    """
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
    """
    Saves the 50 samples of one packet to the active CSV, one row per
    sample. Sample times are spaced 20 ms apart, ending at the time the
    packet was received.

    Author: Bianca Dibden (integration)
    """
    n = len(record.raw)
    fmt = lambda v: "" if v is None else f"{v:.1f}"
    for i in range(n):
        t = record.time - (n - 1 - i) * SAMPLE_PERIOD
        timestamp = datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
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
    """
    Safely closes the active CSV file.
    """
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
    """
    Updates the visual HR alarm based on user-defined limits.
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
# CREATE GUI
# ============================================================

def configure_monitor_grid(window):
    """Allocate Monitor space once; Tk and TkAgg handle subsequent resizing.

    Six equal main tracks allow independent splits: 3:3 for the left plots,
    4:2 for HR/Session on the right. The compact log sits below both columns.
    """
    cells = (
        ("-TREND_PANEL-", 0, 0, 3, 1),
        ("-HR_SIDEBAR-", 0, 1, 4, 1),
        ("-PPG_PANEL-", 3, 0, 3, 1),
        ("-SESSION_PANEL-", 4, 1, 2, 1),
        ("-LOG_PANEL-", 6, 0, 1, 2),
    )
    wrappers = [window[key].ParentRowFrame for key, *_ in cells]
    monitor = wrappers[0].master
    # Convert all sibling wrappers before introducing grid in their parent.
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

    # Expand the real canvas rows inside the plot panels, not only the borders.
    for key in ("-HR_TREND_CANVAS-", "-PPG_CANVAS-"):
        window[key].ParentRowFrame.pack_configure(fill="both", expand=True)


def configure_host_hr_panel(window):
    """Scale only HR/BPM and limits spacing within the existing panel."""
    window.TKroot.update_idletasks()
    panel = window["-HR_SIDEBAR-"].Widget
    number = window["-CURRENT_HR-"].Widget
    bpm = window["-HR_BPM_LABEL-"].Widget
    number_row = window["-CURRENT_HR-"].ParentRowFrame
    limits_row = window["-HR_LIMITS_HEADING-"].ParentRowFrame
    base_width, base_height = panel.winfo_width(), panel.winfo_height()
    row_width = number_row.winfo_reqwidth()
    padding = limits_row.pack_info()["pady"]
    top_pad, bottom_pad = padding if isinstance(padding, tuple) else (padding, padding)
    base_number_height = int(number.tk.call("font", "metrics", number.cget("font"), "-linespace"))
    # Freeze the normal request, not the allocated size, to avoid grid feedback.
    panel.configure(width=panel.winfo_reqwidth(), height=panel.winfo_reqheight())
    panel.pack_propagate(False)
    pending = None
    last_style = (FONT_HR[1], FONT_VALUE[1], 0)

    def apply_scale():
        nonlocal pending, last_style
        pending = None
        scale = max(1.0, min(1.20, panel.winfo_width() / base_width,
                             panel.winfo_height() / base_height,
                             max(0, panel.winfo_width() - 20) / row_width))
        original_number_size = round(FONT_HR[1] * scale)
        number_scale = min(1.0 + 1.5 * (scale - 1.0),
                           max(1.0, (panel.winfo_width() - 20) / row_width))
        number_size = round(FONT_HR[1] * number_scale)
        number_height = int(number.tk.call(
            "font", "metrics", (FONT_HR[0], number_size, *FONT_HR[2:]), "-linespace"))
        # Share added height with the gap, reserving room for every limits row.
        available_gap = max(0, panel.winfo_height() - panel.winfo_reqheight()
                            - max(0, number_height - base_number_height) - 4)
        gap = min(round(max(0, panel.winfo_height() - base_height) * 0.5),
                  available_gap)
        original_height = int(number.tk.call(
            "font", "metrics", (FONT_HR[0], original_number_size, *FONT_HR[2:]), "-linespace"))
        # Keep the limits at their existing position as the number grows.
        original_gap = min(round(max(0, panel.winfo_height() - base_height) * 0.5),
                           max(0, panel.winfo_height() - panel.winfo_reqheight()
                               - max(0, original_height - base_number_height) - 4))
        gap = min(gap, max(0, original_gap - (number_height - original_height)))
        style = (number_size, round(FONT_VALUE[1] * scale), gap)
        if style == last_style:
            return
        if style[0] != last_style[0]:
            number.configure(font=(FONT_HR[0], style[0], *FONT_HR[2:]))
        if style[1] != last_style[1]:
            bpm.configure(font=(FONT_VALUE[0], style[1], *FONT_VALUE[2:]))
        if style[2] != last_style[2]:
            limits_row.pack_configure(pady=(top_pad + style[2], bottom_pad))
        last_style = style

    def schedule_scale(event):
        nonlocal pending
        if pending is None:
            pending = panel.after_idle(apply_scale)

    def cancel_scale(event):
        nonlocal pending
        if event.widget is panel and pending is not None:
            panel.after_cancel(pending)
            pending = None

    panel.bind("<Configure>", schedule_scale, add="+")
    panel.bind("<Destroy>", cancel_scale, add="+")


def configure_session_timer(window):
    """Scale the elapsed-time font independently inside the Session panel."""
    from tkinter import font as tkfont

    window.TKroot.update_idletasks()
    panel = window["-SESSION_PANEL-"].Widget
    timer = window["-SESSION_TIME-"].Widget
    timer_row = window["-SESSION_TIME-"].ParentRowFrame
    base_width, base_height = panel.winfo_width(), panel.winfo_height()
    timer_font = tkfont.Font(root=window.TKroot, font=timer.cget("font"))
    base_size = timer_font.actual("size")
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
        if pending is None:
            pending = panel.after_idle(apply_scale)

    def cancel_scale(event):
        nonlocal pending
        if event.widget is panel and pending is not None:
            panel.after_cancel(pending)
            pending = None

    panel.bind("<Configure>", schedule_scale, add="+")
    panel.bind("<Destroy>", cancel_scale, add="+")


def configure_researcher_grid(window):
    """Graph-first Researcher grid: 78% plots / 22% compact information."""
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


def fit_initial_window(window):
    """Fit and centre once; leave subsequent resizing to the existing grids."""
    root = window.TKroot
    root.update_idletasks()
    wm_width, wm_height = root.maxsize()
    available_width = max(1, min(root.winfo_vrootwidth(), wm_width) - 32)
    available_height = max(1, min(root.winfo_vrootheight(), wm_height) - 32)

    # Plot canvases have comfortable requested sizes, not hard minimums.
    # Base the horizontal minimum on the actual right-side controls instead.
    monitor_width = max(window[key].Widget.winfo_reqwidth()
                        for key in ("-HR_SIDEBAR-", "-SESSION_PANEL-")) / .35
    researcher_width = window["-RESEARCH_SIDEBAR-"].Widget.winfo_reqwidth() / .22
    minimum_width = min(available_width, int(max(monitor_width, researcher_width) + 50))
    minimum_height = min(available_height, root.winfo_reqheight())
    root.minsize(minimum_width, minimum_height)

    width = min(available_width, max(minimum_width, root.winfo_reqwidth()))
    height = min(available_height, max(minimum_height, root.winfo_reqheight()))
    left = root.winfo_vrootx() + 16 + (available_width - width) // 2
    top = root.winfo_vrooty() + 32 + (available_height - height) // 2
    root.geometry(f"{width}x{height}+{left}+{top}")


def create_gui():
    """
    Creates the LumiPulse graphical user interface.
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
            font=("Helvetica", 10, "bold"),
            text_color=RED,
            background_color=BG
        ),
    ]

    # ========================================================
    # CURRENT HR PANEL
    # ========================================================

    current_hr_layout = [
        [
            sg.Text(
                "HOST HEART RATE",
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
                font=("Helvetica", 13),
                size=(7, 1),
                background_color=PANEL
            ),
            sg.Input(
                str(LOW_HR_LIMIT),
                font=("Helvetica", 13),
                size=(7, 1),
                key="-LOW_HR_LIMIT-",
                enable_events=True
            ),
            sg.Text(
                "BPM",
                font=("Helvetica", 13),
                text_color=TEXT_MUTED,
                background_color=PANEL
            )
        ],
        [
            sg.Text(
                "High",
                font=("Helvetica", 13),
                size=(7, 1),
                background_color=PANEL
            ),
            sg.Input(
                str(HIGH_HR_LIMIT),
                font=("Helvetica", 13),
                size=(7, 1),
                key="-HIGH_HR_LIMIT-",
                enable_events=True
            ),
            sg.Text(
                "BPM",
                font=("Helvetica", 13),
                text_color=TEXT_MUTED,
                background_color=PANEL
            )
        ],
        [
            sg.Button(
                "Apply Limits",
                font=("Helvetica", 13),
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
                font=("Helvetica", 11, "bold"),
                size=(28, 2),
                text_color=AMBER,
                background_color=PANEL
            )
        ]
    ]

    # ========================================================
    # RIGHT COLUMN - ROW 1
    # ========================================================

    # One panel: large HR/BPM, then limits, Apply, and textual alarm/status.
    hr_information_column = current_hr_layout + hr_limits_layout

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
                font=("Helvetica", 20, "bold"),
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
    # Each region has its own Tk row wrapper. After finalization these
    # wrappers become the cells of one shared grid (see configure_monitor_grid).
    monitor_layout = [
        [sg.Frame(
            "", hr_trend_layout, key="-TREND_PANEL-",
            background_color=PANEL, border_width=1,
            expand_x=True, expand_y=True
        )],
        [sg.Frame(
            "", hr_information_column, key="-HR_SIDEBAR-",
            background_color=PANEL, border_width=1,
            expand_x=True, expand_y=True
        )],
        [sg.Frame(
            "", ppg_layout, key="-PPG_PANEL-",
            background_color=PANEL, border_width=1,
            expand_x=True, expand_y=True
        )],
        [sg.Frame(
            "", session_layout, key="-SESSION_PANEL-",
            background_color=PANEL, border_width=1,
            expand_x=True, expand_y=True
        )],
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
    # RESEARCHER - PROCESSING INFORMATION
    # ========================================================

    processing_layout = [
        [
            sg.Text(
                "PROCESSING INFORMATION",
                font=FONT_SECTION,
                text_color=TEXT_MUTED,
                background_color=PANEL
            )
        ],
        [
            sg.Text(
                "Sampling rate:",
                size=(18, 1),
                text_color=TEXT_MUTED,
                background_color=PANEL
            ),
            sg.Text(
                "50 Hz",
                background_color=PANEL
            )
        ],
        [
            sg.Text(
                "Host processing:",
                size=(18, 1),
                text_color=TEXT_MUTED,
                background_color=PANEL
            ),
            sg.Text(
                "Waiting for data",
                key="-FILTER_STATUS-",
                background_color=PANEL
            )
        ],
        [
            sg.Text(
                "Last packet:",
                size=(18, 1),
                text_color=TEXT_MUTED,
                background_color=PANEL
            ),
            sg.Text(
                "--",
                key="-LAST_PACKET-",
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
                "Signal quality:",
                size=(18, 1),
                text_color=TEXT_MUTED,
                background_color=PANEL
            ),
            sg.Text(
                "--",
                key="-SIGNAL_QUALITY-",
                size=(34, 2),
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
        ],
        [
            sg.Text(
                "Average HR:",
                size=(18, 1),
                text_color=TEXT_MUTED,
                background_color=PANEL
            ),
            sg.Text(
                "--.- BPM",
                key="-AVERAGE_HR-",
                background_color=PANEL
            )
        ]
    ]

    # Reuse existing measurement elements/keys; only their presentation moves.
    researcher_hr_layout[0][0].DisplayText = "HEART RATE"
    signal_analysis_layout[0][0].DisplayText = "SIGNAL SUMMARY"
    signal_analysis_layout[3][0].DisplayText = "Average Host HR:"
    quality_row = signal_analysis_layout[1]
    quality_value = quality_row[1]
    average_row = signal_analysis_layout[3]
    pulse_value = sg.Text("--", key="-PULSE_COUNT-", visible=False)

    # Keep update targets available to the existing packet/communication code,
    # but remove debugging details from the visible Researcher dashboard.
    filter_value = sg.Text("Waiting for data", key="-FILTER_STATUS-", visible=False)
    last_packet_value = sg.Text("--", key="-LAST_PACKET-", visible=False)
    processing_layout = [
        [sg.Text("PROCESSING", font=FONT_SECTION, text_color=TEXT_MUTED, background_color=PANEL)],
        [sg.Text("Sampling: 50 Hz", background_color=PANEL)],
        [sg.Text("Band-pass: 0.5–5 Hz", background_color=PANEL)],
        [sg.Text("Detection: Adaptive threshold", background_color=PANEL)],
        [sg.Text("Host HR: Median of 3–6 IBIs", background_color=PANEL)],
        [sg.Text("STFT: 4 s Hann, 0.5 s hop", size=(25, 2),
                 background_color=PANEL)],
        [filter_value, last_packet_value, pulse_value]
    ]
    signal_analysis_layout = [
        signal_analysis_layout[0],
        [quality_row[0]], [quality_value], average_row,
        [sg.Text("PPG amplitude (P-P):", text_color=TEXT_MUTED, background_color=PANEL)],
        [sg.Text("--", key="-AMPLITUDE-", background_color=PANEL)],
    ]
    # Fixed display slots preserve stable panel requests during live updates.
    value_sizes = {
        "-MCU_HR-": (10, 1), "-HOST_HR-": (10, 1),
        "-HR_DIFFERENCE-": (10, 1), "-FILTER_STATUS-": (24, 2),
        "-LAST_PACKET-": (24, 1), "-SIGNAL_QUALITY-": (24, 3),
        "-PULSE_COUNT-": (4, 1),
        "-AVERAGE_HR-": (9, 1), "-AMPLITUDE-": (16, 1),
    }
    for panel_layout in (researcher_hr_layout, processing_layout, signal_analysis_layout):
        for row in panel_layout:
            for element in row:
                element.Pad = (2, 2)
                if element.Key in value_sizes:
                    element.Size = value_sizes[element.Key]
                    element.AutoSizeText = False
                elif element.Type == sg.ELEM_TYPE_TEXT:
                    # Keep the long technical caption wrapped inside the sidebar.
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

    def analysis_panel(key):
        return sg.Frame("", [[sg.Canvas(
            key=key, size=(420, 150), background_color=PANEL,
            expand_x=True, expand_y=True
        )]], background_color=PANEL, border_width=1, expand_x=True, expand_y=True)

    # ========================================================
    # RESEARCHER TAB
    # ========================================================

    researcher_layout = [
        [sg.Text("RESEARCHER / SIGNAL ANALYSIS", font=("Helvetica", 16, "bold"),
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

    configure_monitor_grid(window)
    configure_researcher_grid(window)
    fit_initial_window(window)
    configure_host_hr_panel(window)
    configure_session_timer(window)

    return window

# ============================================================

# MAIN PROGRAM

# ============================================================

# ============================================================
# INTEGRATION HELPERS
# ============================================================

def style_legend(ax, handles, *, compact=False):
    """
    Draws the PPG legend above the plot, showing only the given lines.

    Author: Bianca Dibden (integration)
    """
    legend = ax.legend(
        handles=handles,
        loc="upper right" if compact else "lower right",
        bbox_to_anchor=(1.0, 1.15 if compact else 1.01),
        borderaxespad=0,
        borderpad=0.2 if compact else None,
        frameon=True,
        fontsize=9,
        ncol=2
    )
    legend.get_frame().set_facecolor(PANEL)
    legend.get_frame().set_edgecolor(BORDER)
    for legend_text in legend.get_texts():
        legend_text.set_color(TEXT)


def autoscale_y(ax, *arrays, minimum_span=20):
    """
    Fits the y-axis to the finite values in the given arrays, with a 10 %
    margin. Does nothing until real data exists.

    Author: Bianca Dibden (integration)
    """
    values = np.concatenate([a[np.isfinite(a)] for a in arrays])
    if values.size == 0:
        return
    low, high = float(values.min()), float(values.max())
    span = max(high - low, minimum_span)
    middle = (high + low) / 2
    ax.set_ylim(middle - 0.6 * span, middle + 0.6 * span)


def update_hr_display(window, host_hr, signal_message, low_limit, high_limit,
                      signal_ok=True):
    """
    Shows the current host HR to 1 d.p. (Reqts 9, 10) and the HR alarm
    (Reqt 11). If the signal is not trustworthy (host_hr is None) the HR
    is shown as --.- with the reason, and no HR alarm is raised.

    Returns
    -------
    str
        'low', 'high', 'normal' or 'invalid' - used to log alarm changes.

    Author: Bianca Dibden (integration)
    """
    if host_hr is None:
        window["-CURRENT_HR-"].update("--.-", text_color=TEXT_MUTED)
        window["-HR_STATUS-"].update(signal_message.upper(), text_color=AMBER)
        return "invalid"

    window["-CURRENT_HR-"].update(f"{host_hr:.1f}")
    update_hr_status(window, host_hr, low_limit, high_limit)
    if host_hr < low_limit:
        return "low"
    if host_hr > high_limit:
        return "high"
    if not signal_ok:
        # HR is still shown (testing), but flag that it is unverified.
        window["-CURRENT_HR-"].update(text_color=AMBER)
        window["-HR_STATUS-"].update(signal_message.upper(), text_color=AMBER)
    return "normal"

# ============================================================
# MAIN PROGRAM
# ============================================================

def main():
    # Port comes from the command line:  py gui.py COM5
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

    ppg_time = np.linspace(0, PPG_DISPLAY_SECONDS, PPG_BUFFER_SIZE)

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
    ppg_ax.set_xlim(0, PPG_DISPLAY_SECONDS)
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

    hr_times = []
    hr_values = []
    mcu_hr_values = []

    hr_line, = hr_ax.plot([], [], color=GREEN, linewidth=1.5)

    hr_ax.set_xlabel("Time (s)", labelpad=3)
    hr_ax.set_ylabel("Heart Rate (BPM)", labelpad=3)
    hr_ax.set_xlim(0, HR_TREND_SECONDS)
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
    cmp_ax.set_xlim(0, HR_TREND_SECONDS)
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
        """
        Starts or stops CSV recording when the recording state changes.

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
        """
        Updates the HR display/alarm and logs every change of alarm state
        (e.g. 'Pulse Low', Reqt 13), whether caused by new data or by the
        user changing the HR limits.

        Author: Bianca Dibden (integration)
        """
        nonlocal hr_alarm_state
        new_alarm_state = update_hr_display(
            window, host_hr, signal_message, low_hr_limit, high_hr_limit, signal_ok
        )
        if session_summary is not None:
            session_summary.alarm(new_alarm_state)
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

    link = PPGLink(port)
    link.start()

    # ========================================================
    # MAIN EVENT LOOP
    # ========================================================

    while True:

        event, values = window.read(timeout=GUI_POLL_MS)

        if event == sg.WIN_CLOSED:
            break

        now = time.monotonic()
        elapsed_time = now - program_start_time
        # Latch interruptions even when no records arrive in this GUI poll.
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
        records = link.get_records()

        for record in records:
            register_valid_packet(window, record)

            # Recording state from the ESP32 button
            first_sample_time = record.time - (len(record.raw) - 1) * SAMPLE_PERIOD
            set_recording(record.recording)
            if recording_state and csv_writer is not None:
                if recording_start_time is None:
                    recording_start_time = first_sample_time
                save_packet(csv_writer, record, recording_start_time)
                session_summary.observe(record)

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

            # Rolling PPG buffers: drop the oldest 50, add the newest 50
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

        if records:
            # A connection change cannot be located inside a queued batch.
            # Keep its records separate conservatively, then resume continuity.
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
                # Both: remove the raw DC level so both share one axis
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

            if elapsed_time <= HR_TREND_SECONDS:
                display_hr_times = np.array(hr_times)
            else:
                display_hr_times = np.array(hr_times) - (elapsed_time - HR_TREND_SECONDS)

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

            valid_hr = [v for v in hr_values if np.isfinite(v)]
            window["-AVERAGE_HR-"].update(
                f"{np.mean(valid_hr):.1f} BPM" if valid_hr else "--"
            )

        # Independent 1 Hz researcher refresh, including updates after a queued
        # packet burst. Existing Monitor/HR/communication updates retain their cadence.
        if researcher_dirty and now - last_researcher_update >= 1.0:
            update_researcher_plots(researcher_plots, researcher)
            last_researcher_update = now
            researcher_dirty = False
        reports.poll(window)

        # ----------------------------------------------------
        # BLUETOOTH STATUS + NO-DATA ALARM (Reqt 14)
        # ----------------------------------------------------
        update_communication_display(window, communication, link)
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
    # SAFE SHUTDOWN
    # ========================================================

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
