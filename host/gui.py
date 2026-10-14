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

DEVELOPMENT VERSION:

Synthetic data is currently used so the GUI can be developed

before the real Bluetooth data is integrated.

FINAL DATA EXPECTED FROM HOST COMMUNICATION:

- Communication status

- MCU HR

- Recording state

- Raw PPG

- Filtered PPG

- Host HR

"""

# ============================================================
# IMPORTS
# ============================================================

import FreeSimpleGUI as sg
import numpy as np
import time
import csv
import os
from datetime import datetime
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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

# Communication alarm requirement.
COMMUNICATION_TIMEOUT = 5.0

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
    Adds an event to the system log.
    Required format example:
    Thu Sep 19 17:46:50 2024: Pulse Low
    """

    timestamp = datetime.now().strftime(
        "%a %b %d %H:%M:%S %Y"

    )
    window["-SYSTEM_LOG-"].update(
        f"{timestamp}: {message}\n",
        append=True
    )

# ============================================================
# COMMUNICATION STATUS
# ============================================================

def register_valid_packet(window, communication):

    """
    Call this whenever a NEW VALID Bluetooth packet arrives.

    A valid packet:
    - resets the 5-second timeout
    - sets Bluetooth to CONNECTED
    - clears the communication alarm
    """

    communication["last_packet_time"] = time.monotonic()
    packet_time = datetime.now().strftime("%H:%M:%S")
    window["-LAST_PACKET-"].update(packet_time)
    window["-BT_STATUS-"].update(

        "● CONNECTED",
        text_color=GREEN

    )

    # Clear communication-loss alarm when a new packet arrives.

    if communication["lost"]:
        communication["lost"] = False
        window["-COMM_ALARM-"].update("")

        add_log(
            window,
            "Communication restored."

        )

def check_communication(window, communication, now):

    """
    Displays an alarm if no valid packet has arrived for >= 5 s.
    The alarm remains active until another valid packet arrives.
    """

    last_packet_time = communication["last_packet_time"]

    # Do nothing until the first real packet has been received.

    if last_packet_time is None:
        return

    time_since_packet = now - last_packet_time

    if (
        time_since_packet >= COMMUNICATION_TIMEOUT
        and not communication["lost"]

    ):

        communication["lost"] = True
        window["-BT_STATUS-"].update(
            "● DISCONNECTED",
            text_color=RED

        )

        window["-COMM_ALARM-"].update(
            "⚠ COMMUNICATION LOST"
        )

        add_log(
            window,
            "Communication lost."
        )

# ============================================================
# CSV RECORDING
# ============================================================

def start_recording():
    """
    Creates a new CSV file when recording starts.
    """
    os.makedirs(
        "LumiPulse_Recordings",
        exist_ok=True
    )

    file_time = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    filename = os.path.join(
        "LumiPulse_Recordings",
        f"LumiPulse_{file_time}.csv"
    )

    csv_file = open(
        filename,
        "w",
        newline=""
    )
    csv_writer = csv.writer(csv_file)

    # These columns can be expanded when real host data arrives.

    csv_writer.writerow([
        "Timestamp",
        "Session Time (s)",
        "Raw PPG",
        "MCU HR (BPM)",
        "Recording State"
    ])

    return csv_file, csv_writer, filename

def save_sample(
    csv_writer,
    session_time,
    raw_ppg,
    mcu_hr
):

    """
    Saves one sample to the active CSV.
    """

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S.%f"
    )[:-3]

    csv_writer.writerow([
        timestamp,
        f"{session_time:.3f}",
        f"{raw_ppg:.4f}",
        f"{mcu_hr:.1f}",
        "ON"
    ])

def stop_recording(csv_file):

    """
    Safely closes the active CSV file.
    """
    if csv_file is not None:
        csv_file.close()

# ============================================================
# TEMPORARY SYNTHETIC DATA
# ============================================================

def generate_synthetic_hr(t):

    """
    Generates temporary HR data for GUI testing.
    REMOVE/REPLACE when real data is integrated.
    """
    base_hr = 72

    variation = 5 * np.sin(
        2 * np.pi * 0.02 * t
    )
    return base_hr + variation

def generate_synthetic_ppg(t, heart_rate):

    """
    Generates temporary PPG data for GUI testing.
    REMOVE/REPLACE with received Raw PPG later.
    """

    heart_frequency = heart_rate / 60
    pulse = np.sin(
        2 * np.pi * heart_frequency * t
    )

    harmonic = 0.25 * np.sin(
        4 * np.pi * heart_frequency * t
    )

    noise = 0.05 * np.random.randn()
    return pulse + harmonic + noise

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

def main():

    window = create_gui()

    # ========================================================

    # CREATE PPG GRAPH

    # ========================================================

    ppg_fig, ppg_ax, ppg_fig_agg = add_plot(

        window["-PPG_CANVAS-"],

        figsize=(7.0, 2.25)

    )

    ppg_time = np.linspace(

        0,

        PPG_DISPLAY_SECONDS,

        PPG_BUFFER_SIZE

    )

    raw_ppg_buffer = np.zeros(

        PPG_BUFFER_SIZE

    )

    raw_ppg_line, = ppg_ax.plot(

        ppg_time,

        raw_ppg_buffer,

        label="Raw PPG",

        color=ACCENT,

        linewidth=1.5

    )

    ppg_ax.set_xlabel(

        "Time (s)",

        labelpad=3

    )

    ppg_ax.set_ylabel(

        "PPG amplitude (a.u.)",

        labelpad=3

    )

    ppg_ax.set_xlim(

        0,

        PPG_DISPLAY_SECONDS

    )

    ppg_ax.set_ylim(

        -2,

        2

    )

    ppg_ax.grid(

        True,

        alpha=0.2

    )

    # Fixed legend ABOVE the plotting area.

    # It will not move and will not cover the PPG waveform.

    ppg_legend = ppg_ax.legend(

        loc="lower right",

        bbox_to_anchor=(1.0, 1.01),

        borderaxespad=0,

        frameon=True,

        fontsize=9

    )

    ppg_legend.get_frame().set_facecolor(

        PANEL

    )

    ppg_legend.get_frame().set_edgecolor(

        BORDER

    )

    for legend_text in ppg_legend.get_texts():

        legend_text.set_color(TEXT)

    # Smaller margins make graph fill its panel better.

    ppg_fig.subplots_adjust(

        left=0.11,

        right=0.98,

        bottom=0.22,

        top=0.82

    )

    ppg_fig_agg.draw()

    # ========================================================

    # CREATE HR TREND GRAPH

    # ========================================================

    hr_fig, hr_ax, hr_fig_agg = add_plot(

        window["-HR_TREND_CANVAS-"],

        figsize=(7.0, 2.2)

    )

    hr_times = []

    hr_values = []

    hr_line, = hr_ax.plot(

        [],

        [],

        color=GREEN,

        linewidth=1.5

    )

    hr_ax.set_xlabel(

        "Time (s)",

        labelpad=3

    )

    hr_ax.set_ylabel(

        "Heart Rate (BPM)",

        labelpad=3

    )

    hr_ax.set_xlim(

        0,

        HR_TREND_SECONDS

    )

    hr_ax.set_ylim(

        40,

        120

    )

    hr_ax.grid(

        True,

        alpha=0.2

    )

    # Reduce wasted margins so plot fills its frame.

    hr_fig.subplots_adjust(

        left=0.11,

        right=0.98,

        bottom=0.22,

        top=0.96

    )

    hr_fig_agg.draw()

    # ========================================================

    # PROGRAM VARIABLES

    # ========================================================

    sample_number = 0

    program_start_time = time.monotonic()

    next_sample_time = program_start_time

    last_ppg_draw_time = program_start_time

    last_hr_update_time = program_start_time

    current_hr = 72.0

    low_hr_limit = LOW_HR_LIMIT

    high_hr_limit = HIGH_HR_LIMIT

    # ========================================================

    # RECORDING VARIABLES

    # ========================================================

    # TEMPORARY:

    # Test Recording controls this during development.

    #

    # FINAL:

    # recording_state will come from ESP32 packet data.

    recording_state = False

    previous_recording_state = False

    recording_start_time = None

    csv_file = None

    csv_writer = None

    csv_filename = None

    # ========================================================

    # COMMUNICATION VARIABLES

    # ========================================================

    communication = {

        "last_packet_time": None,

        "lost": False

    }

    # ========================================================

    # INITIAL SYSTEM LOG

    # ========================================================

    add_log(

        window,

        "LumiPulse host application started."

    )

    add_log(

        window,

        "Simulation mode active."

    )

    # ========================================================

    # MAIN EVENT LOOP

    # ========================================================

    while True:

        event, values = window.read(

            timeout=5

        )

        if event == sg.WIN_CLOSED:

            break

        now = time.monotonic()

        elapsed_time = (

            now - program_start_time

        )

        # ====================================================

        # CHECK COMMUNICATION

        # ====================================================

        check_communication(

            window,

            communication,

            now

        )

        # FINAL INTEGRATION:

        #

        # When a Bluetooth packet has been received AND

        # successfully validated, call:

        #

        # register_valid_packet(window, communication)

        #

        # If 5 seconds pass without a valid packet,

        # COMMUNICATION LOST is displayed.

        #

        # The next valid packet automatically clears the alarm.

        # ====================================================

        # TEMPORARY TEST RECORDING BUTTON

        # ====================================================

        if event == "-TEST_RECORDING-":

            recording_state = not recording_state

        # ====================================================

        # RECORDING START

        # ====================================================

        if recording_state and not previous_recording_state:

            recording_start_time = now

            (

                csv_file,

                csv_writer,

                csv_filename

            ) = start_recording()

            window["-RECORDING_STATUS-"].update(

                "● ON",

                text_color=RED

            )

            window["-SESSION_RECORDING-"].update(

                "● ON",

                text_color=RED

            )

            window["-SESSION_TIME-"].update(

                "00:00:00"

            )

            add_log(

                window,

                "Recording started."

            )

        # ====================================================

        # RECORDING STOP

        # ====================================================

        elif (

            not recording_state

            and previous_recording_state

        ):

            stop_recording(

                csv_file

            )

            window["-RECORDING_STATUS-"].update(

                "● OFF",

                text_color=TEXT_MUTED

            )

            window["-SESSION_RECORDING-"].update(

                "● OFF",

                text_color=TEXT_MUTED

            )

            add_log(

                window,

                "Recording stopped."

            )

            if csv_filename is not None:

                add_log(

                    window,

                    "CSV saved: "

                    + os.path.basename(csv_filename)

                )

            csv_file = None

            csv_writer = None

            # FUTURE:

            # PDF summary generation can be called here.

        previous_recording_state = recording_state

        # ====================================================

        # SESSION TIMER

        # ====================================================

        if (

            recording_state

            and recording_start_time is not None

        ):

            session_elapsed = (

                now - recording_start_time

            )

            hours = int(

                session_elapsed // 3600

            )

            minutes = int(

                (session_elapsed % 3600) // 60

            )

            seconds = int(

                session_elapsed % 60

            )

            window["-SESSION_TIME-"].update(

                f"{hours:02d}:"

                f"{minutes:02d}:"

                f"{seconds:02d}"

            )

        # ====================================================

        # TEMPORARY SYNTHETIC PPG DATA

        # ====================================================

        while now >= next_sample_time:

            sample_time = (

                sample_number / SAMPLE_RATE

            )

            synthetic_hr = generate_synthetic_hr(

                sample_time

            )

            new_ppg_sample = generate_synthetic_ppg(

                sample_time,

                synthetic_hr

            )

            # ================================================

            # SAVE SAMPLE WHILE RECORDING

            # ================================================

            if (

                recording_state

                and csv_writer is not None

            ):

                session_time = (

                    now - recording_start_time

                )

                save_sample(

                    csv_writer,

                    session_time,

                    new_ppg_sample,

                    synthetic_hr

                )

            # ================================================

            # UPDATE PPG BUFFER

            # ================================================

            raw_ppg_buffer[:-1] = (

                raw_ppg_buffer[1:]

            )

            raw_ppg_buffer[-1] = (

                new_ppg_sample

            )

            sample_number += 1

            next_sample_time += (

                SAMPLE_PERIOD

            )

        # ====================================================

        # REDRAW PPG GRAPH

        # ====================================================

        if (

            now - last_ppg_draw_time

            >= PPG_PLOT_PERIOD

        ):

            raw_ppg_line.set_ydata(

                raw_ppg_buffer

            )

            ppg_fig_agg.draw()

            last_ppg_draw_time = now

        # ====================================================

        # UPDATE HR DISPLAY AT 1 Hz

        # ====================================================

        if (

            now - last_hr_update_time

            >= HR_UPDATE_PERIOD

        ):

            # TEMPORARY:

            # Replace with received MCU/Host HR later.

            current_hr = generate_synthetic_hr(

                elapsed_time

            )

            displayed_hr = (

                f"{current_hr:.1f}"

            )

            # ================================================

            # CURRENT HR

            # ================================================

            window["-CURRENT_HR-"].update(

                displayed_hr

            )

            # ================================================

            # RESEARCHER MCU HR

            # ================================================

            window["-MCU_HR-"].update(

                f"{displayed_hr} BPM"

            )

            # ================================================

            # HR STATUS

            # ================================================

            update_hr_status(

                window,

                current_hr,

                low_hr_limit,

                high_hr_limit

            )

            # ================================================

            # STORE HR TREND

            # ================================================

            hr_times.append(

                elapsed_time

            )

            hr_values.append(

                current_hr

            )

            # Keep only most recent 60 seconds.

            while (

                len(hr_times) > 0

                and

                elapsed_time - hr_times[0]

                > HR_TREND_SECONDS

            ):

                hr_times.pop(0)

                hr_values.pop(0)

            # ================================================

            # ROLLING 0-60 SECOND HR DISPLAY

            # ================================================

            if len(hr_times) > 0:

                if elapsed_time <= HR_TREND_SECONDS:

                    display_hr_times = np.array(

                        hr_times

                    )

                else:

                    window_start = (

                        elapsed_time

                        - HR_TREND_SECONDS

                    )

                    display_hr_times = (

                        np.array(hr_times)

                        - window_start

                    )

                hr_line.set_data(

                    display_hr_times,

                    hr_values

                )

            hr_ax.set_xlim(

                0,

                HR_TREND_SECONDS

            )

            hr_fig_agg.draw()

            # ================================================

            # AVERAGE HR

            # ================================================

            if len(hr_values) > 0:

                average_hr = np.mean(

                    hr_values

                )

                window["-AVERAGE_HR-"].update(

                    f"{average_hr:.1f} BPM"

                )

            last_hr_update_time = now

        # ====================================================

        # APPLY USER-DEFINED HR LIMITS

        # ====================================================

        if event == "-APPLY_LIMITS-":

            try:

                new_low_limit = float(

                    values["-LOW_HR_LIMIT-"]

                )

                new_high_limit = float(

                    values["-HIGH_HR_LIMIT-"]

                )

                if (

                    new_low_limit

                    >= new_high_limit

                ):

                    add_log(

                        window,

                        "Invalid HR limits: "

                        "low must be below high."

                    )

                else:

                    low_hr_limit = (

                        new_low_limit

                    )

                    high_hr_limit = (

                        new_high_limit

                    )

                    add_log(

                        window,

                        "HR limits updated: "

                        f"{low_hr_limit:.1f}-"

                        f"{high_hr_limit:.1f} BPM."

                    )

                    update_hr_status(

                        window,

                        current_hr,

                        low_hr_limit,

                        high_hr_limit

                    )

            except ValueError:

                add_log(

                    window,

                    "Invalid HR limits: "

                    "enter numeric values."

                )

        # ====================================================

        # PPG DISPLAY SELECTION

        # ====================================================

        if event == "-SHOW_RAW-":

            raw_ppg_line.set_visible(

                True

            )

            ppg_fig_agg.draw()

            add_log(

                window,

                "PPG display changed to Raw."

            )

        if event == "-SHOW_FILTERED-":

            # FINAL:

            # Show received filtered PPG here.

            add_log(

                window,

                "Filtered PPG selected - "

                "filtered data not yet available."

            )

        if event == "-SHOW_BOTH-":

            # FINAL:

            # Show Raw + Filtered PPG here.

            add_log(

                window,

                "Raw + Filtered selected - "

                "filtered data not yet available."

            )

    # ========================================================

    # SAFE SHUTDOWN

    # ========================================================

    if csv_file is not None:

        stop_recording(csv_file)

    window.close()

# ============================================================

# RUN PROGRAM

# ============================================================

if __name__ == "__main__":

    main()
