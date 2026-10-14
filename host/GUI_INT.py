
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
from datetime import datetime
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from comms import PPGLink, STATUS_CONNECTED, STATUS_CONNECTING
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
FONT_SMALL = ("Helvetica", 9)
FONT_HR = ("Helvetica", 38, "bold")
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

# Fixed width for BOTH right-side columns.

# This keeps Session aligned with Current HR / HR Limits.
SIDE_PANEL_WIDTH = 220

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

    fig_agg = FigureCanvasTkAgg(
        fig,
        canvas_element.TKCanvas

    )

    fig_agg.draw()
    fig_agg.get_tk_widget().pack(
        side="top",
        fill="both",
        expand=True

    )

    return fig, ax, fig_agg

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

    # Never leave a stale HR on screen while there is no data.
    # The next valid packet overwrites this.
    if link.no_data_alarm:
        window["-CURRENT_HR-"].update("--.-", text_color=TEXT_MUTED)
        window["-HR_STATUS-"].update("NO DATA - CHECK CONNECTION", text_color=RED)


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

    csv_file = open(filename, "w", newline="")
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

        sg.VSeparator(),
        sg.Text(
            "Recording:",
            text_color=TEXT_MUTED,
            background_color=BG
        ),

        sg.Text(
            "● OFF",
            key="-RECORDING_STATUS-",
            text_color=TEXT_MUTED,
            background_color=BG
        )
    ]

    # ========================================================
    # CURRENT HR PANEL
    # ========================================================

    current_hr_layout = [

        [
            sg.Text(
                "CURRENT HEART RATE",
                font=FONT_SECTION,
                text_color=TEXT_MUTED,
                background_color=PANEL
            )
        ],

        [
            sg.Text(
                "--.-",
                key="-CURRENT_HR-",
                font=FONT_HR,
                text_color=GREEN,
                background_color=PANEL
            ),

            sg.Text(
                "BPM",
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
                font=FONT_SECTION,
                text_color=TEXT_MUTED,
                background_color=PANEL
            )

        ],

        [
            sg.Text(
                "Low",
                size=(7, 1),
                background_color=PANEL
            ),

            sg.Input(
                str(LOW_HR_LIMIT),
                size=(7, 1),
                key="-LOW_HR_LIMIT-"
            ),

            sg.Text(
                "BPM",
                text_color=TEXT_MUTED,
                background_color=PANEL
            )

        ],

        [
            sg.Text(
                "High",
                size=(7, 1),
                background_color=PANEL
            ),

            sg.Input(
                str(HIGH_HR_LIMIT),
                size=(7, 1),
                key="-HIGH_HR_LIMIT-"
            ),

            sg.Text(
                "BPM",
                text_color=TEXT_MUTED,
                background_color=PANEL
            )
        ],

        [
            sg.Button(
                "Apply Limits",
                key="-APPLY_LIMITS-",
                size=(12, 1)
            )

        ],

        [
            sg.Text(
                "WAITING FOR HEART RATE",
                key="-HR_STATUS-",
                font=("Helvetica", 11, "bold"),
                text_color=AMBER,
                background_color=PANEL
            )
        ]

    ]

    # ========================================================
    # RIGHT COLUMN - ROW 1
    # ========================================================

    hr_information_column = [

        [
            sg.Frame(
                "",
                current_hr_layout,
                background_color=PANEL,
                border_width=1,
                size=(SIDE_PANEL_WIDTH, 105)
            )

        ],

        [

            sg.Frame(

                "",
                hr_limits_layout,
                background_color=PANEL,
                border_width=1,
                size=(SIDE_PANEL_WIDTH, 155)
            )

        ]

    ]

    # ========================================================
    # HEART RATE TREND PANEL
    # ========================================================

    hr_trend_layout = [

        [

            sg.Text(
                "HEART RATE TREND",
                font=FONT_SECTION,
                text_color=TEXT_MUTED,
                background_color=PANEL
            )

        ],

        [

            sg.Canvas(
                key="-HR_TREND_CANVAS-",
                size=(690, 215),
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
                size=(690, 220),
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

        ],

        [

            sg.HorizontalSeparator()

        ],

        [

            # TEMPORARY DEVELOPMENT BUTTON.

            # Remove when recording state comes from ESP32.

            sg.Button(

                "Test Recording",

                key="-TEST_RECORDING-",

                size=(17, 1)

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

                size=(110, 5),

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

    monitor_layout = [

        # ROW 1: HR trend + HR information

        [

            sg.Frame(

                "",

                hr_trend_layout,

                background_color=PANEL,

                border_width=1,

                expand_x=True

            ),

            sg.Column(

                hr_information_column,

                background_color=BG,

                vertical_alignment="top",

                size=(SIDE_PANEL_WIDTH, 270)

            )

        ],

        # ROW 2: PPG + Session

        [

            sg.Frame(

                "",

                ppg_layout,

                background_color=PANEL,

                border_width=1,

                expand_x=True

            ),

            sg.Frame(

                "",

                session_layout,

                background_color=PANEL,

                border_width=1,

                vertical_alignment="top",

                size=(SIDE_PANEL_WIDTH, 285)

            )

        ],

        # ROW 3: System log

        [

            sg.Frame(

                "",

                system_log_layout,

                background_color=PANEL,

                border_width=1,

                expand_x=True

            )

        ]

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

                size=(850, 280),

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

    # ========================================================

    # RESEARCHER TAB

    # ========================================================

    researcher_layout = [

        [

            sg.Text(

                "RESEARCHER / SIGNAL ANALYSIS",

                font=("Helvetica", 16, "bold"),

                text_color=ACCENT,

                background_color=BG

            )

        ],

        [

            sg.Text(

                "Technical comparison of microcontroller and "

                "host-computer signal processing.",

                text_color=TEXT_MUTED,

                background_color=BG

            )

        ],

        [

            sg.Frame(

                "",

                researcher_hr_layout,

                background_color=PANEL,

                border_width=1

            ),

            sg.Frame(

                "",

                processing_layout,

                background_color=PANEL,

                border_width=1

            )

        ],

        [

            sg.Frame(

                "",

                comparison_plot_layout,

                background_color=PANEL,

                border_width=1

            )

        ],

        [

            sg.Frame(

                "",

                signal_analysis_layout,

                background_color=PANEL,

                border_width=1,

                expand_x=True

            )

        ]

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

    return window

# ============================================================

# MAIN PROGRAM

# ============================================================

# ============================================================
# INTEGRATION HELPERS
# ============================================================

def style_legend(ax, handles):
    """
    Draws the PPG legend above the plot, showing only the given lines.

    Author: Bianca Dibden (integration)
    """
    legend = ax.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.01),
        borderaxespad=0,
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


def update_hr_display(window, host_hr, signal_message, low_limit, high_limit):
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
    ppg_ax.set_ylabel("Raw PPG (ADC counts)", labelpad=3)
    ppg_ax.set_xlim(0, PPG_DISPLAY_SECONDS)
    ppg_ax.set_ylim(0, 4095)
    ppg_ax.grid(True, alpha=0.2)
    style_legend(ppg_ax, [raw_ppg_line])

    ppg_fig.subplots_adjust(left=0.11, right=0.98, bottom=0.22, top=0.82)
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
    hr_ax.set_ylim(40, 120)
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
    cmp_ax.set_ylim(40, 120)
    cmp_ax.grid(True, alpha=0.2)
    style_legend(cmp_ax, [cmp_host_line, cmp_mcu_line])
    cmp_fig.subplots_adjust(left=0.09, right=0.98, bottom=0.2, top=0.85)
    cmp_fig_agg.draw()

    # ========================================================
    # PROGRAM VARIABLES
    # ========================================================

    program_start_time = time.monotonic()
    low_hr_limit = LOW_HR_LIMIT
    high_hr_limit = HIGH_HR_LIMIT
    current_hr = None
    latest_record = None
    hr_alarm_state = None
    pulse_count = 0

    # ========================================================
    # RECORDING VARIABLES
    # ========================================================
    # recording_state = hardware button (from the ESP32 packet)
    # OR the on-screen Test Recording button.

    hardware_recording = False
    manual_recording = False
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
        nonlocal csv_file, csv_writer, csv_filename

        if new_state and not recording_state:
            recording_start_monotonic = time.monotonic()
            # CSV session time starts at the first saved sample (set below)
            recording_start_time = None
            csv_file, csv_writer, csv_filename = start_recording()
            window["-RECORDING_STATUS-"].update("● ON", text_color=RED)
            window["-SESSION_RECORDING-"].update("● ON", text_color=RED)
            window["-SESSION_TIME-"].update("00:00:00")
            add_log(window, "Recording started.")

        elif not new_state and recording_state:
            stop_recording(csv_file)
            window["-RECORDING_STATUS-"].update("● OFF", text_color=TEXT_MUTED)
            window["-SESSION_RECORDING-"].update("● OFF", text_color=TEXT_MUTED)
            add_log(window, "Recording stopped.")
            if csv_filename is not None:
                add_log(window, "CSV saved: " + os.path.basename(csv_filename))
            csv_file = None
            csv_writer = None
            # FUTURE: PDF summary - ppg_dsp.analyse_recording() can analyse
            # the raw column of the saved CSV here.

        recording_state = new_state

    def show_hr(host_hr, signal_message):
        """
        Updates the HR display/alarm and logs every change of alarm state
        (e.g. 'Pulse Low', Reqt 13), whether caused by new data or by the
        user changing the HR limits.

        Author: Bianca Dibden (integration)
        """
        nonlocal hr_alarm_state
        new_alarm_state = update_hr_display(
            window, host_hr, signal_message, low_hr_limit, high_hr_limit
        )
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
        # TEST RECORDING BUTTON
        # ----------------------------------------------------
        if event == "-TEST_RECORDING-":
            manual_recording = not manual_recording
            set_recording(hardware_recording or manual_recording)

        # ----------------------------------------------------
        # NEW PACKETS (normally one per second)
        # ----------------------------------------------------
        records = link.get_records()

        for record in records:
            register_valid_packet(window, record)

            # Recording state from the ESP32 button
            hardware_recording = record.recording
            first_sample_time = record.time - (len(record.raw) - 1) * SAMPLE_PERIOD
            set_recording(hardware_recording or manual_recording)
            if recording_state and csv_writer is not None:
                if recording_start_time is None:
                    recording_start_time = first_sample_time
                save_packet(csv_writer, record, recording_start_time)

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

            # beat_times is provided by newer versions of comms.py
            beat_times = getattr(record, "beat_times", None)
            if beat_times is not None:
                pulse_count += len(beat_times)
            latest_record = record

        if records:
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
                raw_centred = raw_ppg_buffer - np.nanmean(raw_ppg_buffer)
                raw_ppg_line.set_ydata(raw_centred)
                filtered_ppg_line.set_ydata(filtered_ppg_buffer)
                autoscale_y(ppg_ax, raw_centred, filtered_ppg_buffer)
            ppg_fig_agg.draw()

            # ------------------------------------------------
            # CURRENT HR + ALARM (Reqts 9, 10, 11)
            # ------------------------------------------------
            show_hr(rec.host_hr, rec.signal_message)

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
            cmp_fig_agg.draw()

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
            if hasattr(rec, "beat_times"):
                window["-PULSE_COUNT-"].update(str(pulse_count))

            valid_hr = [v for v in hr_values if np.isfinite(v)]
            if valid_hr:
                window["-AVERAGE_HR-"].update(f"{np.mean(valid_hr):.1f} BPM")

        # ----------------------------------------------------
        # BLUETOOTH STATUS + NO-DATA ALARM (Reqt 14)
        # ----------------------------------------------------
        update_communication_display(window, communication, link)

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
                if new_low_limit >= new_high_limit:
                    add_log(window, "Invalid HR limits: low must be below high.")
                else:
                    low_hr_limit = new_low_limit
                    high_hr_limit = new_high_limit
                    add_log(
                        window,
                        f"HR limits updated: {low_hr_limit:.1f}-{high_hr_limit:.1f} BPM."
                    )
                    if latest_record is not None and not link.no_data_alarm:
                        show_hr(current_hr, latest_record.signal_message)
            except ValueError:
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
                raw_ppg_line.set_ydata(raw_ppg_buffer)
                ppg_ax.set_ylabel("Raw PPG (ADC counts)", labelpad=3)
                style_legend(ppg_ax, [raw_ppg_line])
                autoscale_y(ppg_ax, raw_ppg_buffer)
            elif ppg_display_mode == "filtered":
                filtered_ppg_line.set_ydata(filtered_ppg_buffer)
                ppg_ax.set_ylabel("Filtered PPG (counts)", labelpad=3)
                style_legend(ppg_ax, [filtered_ppg_line])
                autoscale_y(ppg_ax, filtered_ppg_buffer)
            else:
                raw_centred = raw_ppg_buffer - (
                    np.nanmean(raw_ppg_buffer) if np.isfinite(raw_ppg_buffer).any() else 0
                )
                raw_ppg_line.set_ydata(raw_centred)
                filtered_ppg_line.set_ydata(filtered_ppg_buffer)
                ppg_ax.set_ylabel("PPG (counts, raw DC removed)", labelpad=3)
                style_legend(ppg_ax, [raw_ppg_line, filtered_ppg_line])
                autoscale_y(ppg_ax, raw_centred, filtered_ppg_buffer)

            ppg_fig_agg.draw()
            add_log(window, f"PPG display changed to {ppg_display_mode.capitalize()}.")

    # ========================================================
    # SAFE SHUTDOWN
    # ========================================================

    link.stop()
    if csv_file is not None:
        stop_recording(csv_file)
    window.close()


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()
