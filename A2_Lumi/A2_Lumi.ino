/*
** BMET2922 - LumiPulse (local monitor + Bluetooth SPP link to the host GUI)
**
** Structure: Lab W3 scheduler, latchingSwitch and calcPulse examples.
** Filter/percentiles: adapted from our previous bandpassadapt sketch.
** Sensor and button sampled every 20 ms (50 Hz), debounce count 5.
** Short press: toggle recording flag / blue LED on RELEASE.
** Hold for 3 seconds: toggle buzzer sound OFF/ON without changing recording.
** Sound starts enabled at boot; only another long hold changes it.
** LCD bottom-right ! means buzzer sound is disabled.
** Recording is a flag sent to the host; no samples are saved here.
**
** Bluetooth: device name DEVICE_NAME. Every 50 samples (1 s) one fixed-length
** text packet (sequence, MCU HR, recording flag, 50 raw samples, checksum)
** is sent. It is built by buildPacket() in packet.cpp, which matches
** host/packet_format.py exactly (268 bytes, see packet.h). A separate
** FreeRTOS task does the sending so a slow Bluetooth write can never delay
** the 20 ms sampling loop. The host sends back "$LIM,LLL.L,HHH.H*CC" (the
** low/high HR limits chosen in the GUI) on change and once a second; these
** replace the default alarm limits until the next valid message.
** LCD row 2 shows the connection state.
**
** Wiring: PPG power -> 3V3, ground -> GND, signal -> GPIO34.
** Button -> GPIO33 and GND (internal pull-up, active LOW).
** Red LED -> GPIO17; blue LED -> GPIO26; passive buzzer -> GPIO27.
** LCD: RS=23, E=22, D4=21, D5=19, D6=18, D7=25, RW=GND.
** Keep switch.h, switch.cpp, packet.h and packet.cpp beside this sketch.
*/

// ---------------- LIBRARIES ----------------

// Arduino functions
#include <Arduino.h>
// Library to control the LCD
#include <LiquidCrystal.h> 
// Math functions
#include <math.h>
// Memory management and other C utilities
#include <stdlib.h>
// C character functions
#include <string.h>
// Bluetooth Classic Serial communications
#include "BluetoothSerial.h"
// FreeRTOS, allows several tasks to run concurrently
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
// Provided button/debounce
#include "switch.h"
// Custom for packaging data during the bluetooth communication
#include "packet.h"

// ---------------- PINS AND SETTINGS ----------------

#define RED_LED_PIN 17
#define BLUE_LED_PIN 26
#define BUTTON_PIN 33
#define HR_SENSOR_PIN 34
#define BUZZER_PIN 27
#define DEBOUNCE_COUNT 5

#define LCD_RS_PIN 23
#define LCD_ENABLE_PIN 22
#define LCD_D4_PIN 21
#define LCD_D5_PIN 19
#define LCD_D6_PIN 18
#define LCD_D7_PIN 25

// Init LCD screen connections pins
LiquidCrystal lcd(LCD_RS_PIN, LCD_ENABLE_PIN, LCD_D4_PIN, LCD_D5_PIN, LCD_D6_PIN, LCD_D7_PIN);
// Init the button debounce/state tracker, passing the id=BUTTON_PIN, and the debounce count value=DEBOUNCE_COUNT, both set earlier.
Switch button(BUTTON_PIN, DEBOUNCE_COUNT); // REQUIREMENT #3 PROOF. Using provided lib.

const uint32_t BUTTON_MUTE_HOLD_MS = 3000UL;
const uint32_t LED_FLASH_MS = 100UL;
const uint32_t LCD_MS = 250UL;
const uint16_t BUZZER_HZ = 2000;
// Defining the HB sensor sampling interval, in µs, so we have a greater precission. Requirements say to use 20ms.
constexpr uint32_t SAMPLE_INTERVAL_US = 20000UL;
// Sample frequency used by the signal-processing filters, calculated by dividing the 1,000,000 microseconds per second, by the SAMPLE_INTERVAL_US. Requirements make us use 50, as 1000000/20000 = 50. 
// It tells the high-pass and low-pass filters that samples arrive at 50 Hz.
constexpr float SAMPLE_HZ = 1000000.0f / SAMPLE_INTERVAL_US;

// Cutoff frequencies of the digital FILTERS applied to the heart-rate sensor signal. This corresponds approx to 0.5 Hz × 60 = 30 BPM / 5 Hz × 60 = 300 BPM
const float HIGH_PASS_CUTOFF_HZ = 0.5f; // Reduces slow baseline drift and gradual changes in finger pressure.
const float LOW_PASS_CUTOFF_HZ = 5.0f; // Reduces fast electrical and sensor noise.

// Heartrate values that trigger alarm
const float LOW_HR_ALARM_THRESHOLD_BPM = 50.0f;
const float HIGH_HR_ALARM_THRESHOLD_BPM = 100.0f;

// Measurement limits: 40-150 BPM (because 60,000 / 400  = 150 BPM and 60,000 / 1,500 = 40 BPM). 
// Valid beat-interval range used to reject implausible detections. Alarm limits above are independent.
const uint32_t MIN_VALID_BEAT_INTERVAL_MS = 400UL;
const uint32_t MAX_VALID_BEAT_INTERVAL_MS = 1500UL;

// Two seconds, to settle the filter, then two-second percentile windows.
constexpr uint16_t FILTER_WARMUP_SAMPLE_COUNT = static_cast<uint16_t>(SAMPLE_HZ * 2.0f);
constexpr uint16_t THRESHOLD_WINDOW_SAMPLE_COUNT = static_cast<uint16_t>(SAMPLE_HZ * 2.0f);

// Provisional filtered ADC range: calibrate with real sensor recordings.
// This amplitude check alone cannot identify all movement / bad contact.
const float MIN_VALID_FILTERED_RANGE = 100.0f; // Checks whether the filtered waveform has enough variation to be considered usable.
const float THRESHOLD_HYSTERESIS_FRACTION = 0.05f; // Separates the upper and lower thresholds so small noise fluctuations do not create repeated beat detections.
bool is_signal_amplitude_sufficient = false; // Tells the program whether the filtered waveform has enough variation to pass the amplitude check.

// Clears an old HR after 1.7 s without an accepted beat.
// Slightly longer than the slowest accepted beat interval (1.5 s) declared above as MAX_VALID_BEAT_INTERVAL_MS.
const uint32_t NO_BEAT_TIMEOUT_MS = MAX_VALID_BEAT_INTERVAL_MS + 200UL;

// ---------------- BLUETOOTH SETTINGS ----------------
// Init SerialBT for bluetooth data exchange
BluetoothSerial SerialBT;
// BT Device Name searched by the HOST
const char* DEVICE_NAME = "AbiESP32";

bool is_bluetooth_ready = false;

// Defines the format of one packet of data held in memory.
struct BluetoothPacket {
  uint32_t sequence_number;    // The packet sequence number. It lets the host identify packet order and detect missing packets. Cycles from 0 to 65,535, the hosts wraps/modulo it. 
  float hr_in_bpm;             // Calculated HR in BPM; 0 means no valid measurement.
  uint8_t recording;           // 0 = recording off, 1 = recording on (formatted as ASCII later).
  uint16_t raw_ppg_samples[SAMPLES_PER_PACKET]; // Stores the raw sensor samples included in the packet, each packet contains SAMPLES_PER_PACKET (packet.h) raw sensor readings.
};

// The Bluetooth system should work like this: sensor readings → temporary packet → queue → Bluetooth sender
// One part of the program collects sensor data. Another part sends Bluetooth data. The queue lets those two parts communicate safely.
// bluetooth_packet_queue represents a mailbox provided by FreeRTOS. It will be used to pass completed packets to the Bluetooth task.
QueueHandle_t bluetooth_packet_queue = nullptr; // The "mailbox" has not been created yet
BluetoothPacket packet_being_filled = {}; // This creates a variable that can hold one complete packet, basde on the previous format, the {} sets everything to zero as default. 
// The program reuses this object while collecting sensor readings.
// sample 1  → packet_being_filled.raw_ppg_samples[0]
// sample 2  → packet_being_filled.raw_ppg_samples[1]
// Once all 50 samples have been collected, the packet is complete.

uint16_t samples_in_packet = 0; // Count of how many samples have currently been stored in packet_being_filled
uint32_t next_packet_sequence_number = 0; // Sequence number that will be assigned to the next completed packet.

// ---------------- ALARM LIMITS ----------------
// These values can be overwritten by the HOST, defaults initially to constant values, defined previously.
float low_hr_limit_bpm = LOW_HR_ALARM_THRESHOLD_BPM;
float high_hr_limit_bpm = HIGH_HR_ALARM_THRESHOLD_BPM;
char host_alarm_limits_line[24] = {}; // Capacity: 24 characters. 
// The 24 is the buffer’s capacity, our host alarm-limit message has this format $LIM,050.0,100.0*CC and has a length of 19 characters.
// The buffer needs additional room for: - a possible newline character: \n - the C-string terminator: '\0' - a small safety margin
uint8_t host_alarm_limits_line_length = 0; // Current content: 0 characters

// ---------------- SCHEDULER AND BUTTON SETTINGS ----------------
// Store timing information (decides when each repeated task should run) and the current button, recording, and buzzer states.
// every 20 ms  → sample sensor and update button
// every 250 ms → update LCD
// every 20 ms → check Bluetooth input
uint32_t last_sample_time_us = 0; // Time of the previous sampling/scheduler cycle.
uint32_t last_lcd_update_time_ms = 0; // Time of the previous LCD update.
bool is_recording_active = false; // Whether recording is currently active.
bool is_button_hold_action_triggered = false; // Prevents one continuous long press from repeatedly triggering the hold action.
uint32_t button_press_start_time_ms = 0; // Time at which the current button press began.
bool is_buzzer_enabled = false; // Whether the buzzer is permitted to sound. Muted for now (LCD shows !); hold the button 3 s to enable.
bool is_buzzer_currently_on = false; // Whether the buzzer output is currently active.

// ---------------- FILTER AND THRESHOLD SETTINGS ----------------
// The sensor produces a stream of numbers representing the light detected through the finger.
// The heartbeat creates a small repeating wave, but the readings also contain a large baseline and unwanted fluctuations.

// 1. Raw sensor reading
// 2. High-pass filter: reduce slow baseline changes
// 3. Low-pass filter: reduce rapid fluctuations
// 4. Adaptive thresholds: decide when a pulse occurs
// 5. Time between pulses → heart rate

// Filter coefficients calculated in setup() from the sampling frequency and cutoff frequencies. These are multipliers used in the filter calculations.
float high_pass_coefficient_b = 0.0f;
float high_pass_coefficient_a = 0.0f;
float low_pass_coefficient_b = 0.0f;
float low_pass_coefficient_a = 0.0f;

// Previous values required by the recursive digital filters. This give the filters a small amount of memory.
float previous_raw_sample = 0.0f;
float previous_high_pass_output = 0.0f;
float previous_low_pass_output = 0.0f;
bool is_filter_initialized = false;

uint16_t filter_warmup_sample_count = 0; // Counts readings while the filters settle. 2 seconds by default, so 100 samples.
float threshold_sample_window[THRESHOLD_WINDOW_SAMPLE_COUNT] = {}; // holds 100 filtered readings.
float sorted_threshold_sample_window[THRESHOLD_WINDOW_SAMPLE_COUNT] = {}; // holds a copy sorted from smallest to largest.
uint16_t threshold_window_sample_count = 0; // counts how many readings have been collected.
// Sorting lets the program calculate P10 and P90:
// - P10: approximately 10% of the readings are below this value.
// - P90: approximately 90% are below this value.

// NOTE: Because collection starts after warm-up, the first threshold becomes available after approximately four seconds: two seconds warming up, then two seconds collecting a window.

// Current adaptive threshold and its hysteresis limits. It's adaptive because the program recalculates it from the waveform.
float adaptive_threshold = 0.0f; 
float upper_beat_threshold = 0.0f;
float lower_beat_threshold = 0.0f;
bool is_adaptive_threshold_ready = false;
bool is_beat_detector_armed = false;
float previous_filtered_sample = 0.0f;


// ---------------- HEART RATE AND LED DATA ----------------
bool has_previous_beat = false;
uint32_t last_beat_time_ms = 0;
float heart_rate_bpm = 0;
// The host packet carries 0.0 when this is false (no valid pulse).
// Raw PPG samples are still sent when HR is invalid.
bool is_heart_rate_valid = false;
bool is_red_led_on = false;
uint32_t red_led_flash_start_time_ms = 0;
char previous_lcd_rows[2][17] = {{0}, {0}};

// Helpers keep each scheduled task easy to follow.
bool collect_threshold_sample(float filtered_sample);
void print_debug_data(uint16_t raw_sample, float filtered_sample);
void update_beat_led(uint32_t current_time_ms);
void startup_sequence();
void update_button(uint32_t current_time_ms);
float band_pass_filter(float raw_sample);
int compare_floats(const void* a, const void* b);
float percentile(float fraction);
void update_thresholds();
void reset_beats();
void detect_beat(float filtered_sample, bool thresholds_changed, uint32_t current_time_ms);
void update_alarm();
void read_host_limits();
void write_lcd_row(uint8_t row, const char* text);
void update_lcd();
void buffer_bluetooth_sample(uint16_t raw_sample);
void bluetooth_sender(void* parameter);

void setup() {
  // Init pins 
  pinMode(RED_LED_PIN, OUTPUT);
  pinMode(BLUE_LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(HR_SENSOR_PIN, INPUT);
  digitalWrite(RED_LED_PIN, LOW);
  digitalWrite(BLUE_LED_PIN, LOW);
  digitalWrite(BUZZER_PIN, LOW);
  Serial.begin(115200);
  analogReadResolution(12); // means the raw reading is represented by values from 0 to 4095. These are ADC counts, not directly volts. analog ≠ digital
  // Init analog (HR Sensor)
  analogSetPinAttenuation(HR_SENSOR_PIN, ADC_11db);

  // Same first-order high-pass + low-pass as bandpassadapt.
  // HIGH_PASS_CUTOFF_HZ and LOW_PASS_CUTOFF_HZ are nominal cutoffs, not sharp frequency boundaries.
  float k = tanf(PI * HIGH_PASS_CUTOFF_HZ / SAMPLE_HZ);
  high_pass_coefficient_b = 1.0f / (1.0f + k);
  high_pass_coefficient_a = (k - 1.0f) / (k + 1.0f);
  k = tanf(PI * LOW_PASS_CUTOFF_HZ / SAMPLE_HZ);
  low_pass_coefficient_b = k / (1.0f + k);
  low_pass_coefficient_a = (k - 1.0f) / (k + 1.0f);

  lcd.begin(16, 2); // Set screen dimensions
  startup_sequence(); // <-- STARTUP PROCESS, with text, sound and lights.

  // Start Bluetooth once per boot, after the startup delays.
  bluetooth_packet_queue = xQueueCreate(1, sizeof(BluetoothPacket)); // Creates a FreeRTOS queue, that can hold 1 completed packet + info on the number of bytes needed to store that packet.
  
  // If creation succeeds, the function returns a handle identifying the queue. If it fails, it returns nullptr.
  if (bluetooth_packet_queue != nullptr && SerialBT.begin(DEVICE_NAME)) {
    is_bluetooth_ready = true;
    // Asks FreeRTOS to run your Bluetooth sender as another separate task. 
    // A Bluetooth write can take time. Putting it in a separate task allows the sampling code to continue while the sender waits. 
    // It reduces interference with sampling, although it does not guarantee perfect timing.
    // | Argument | Meaning |
    // | `bluetooth_sender` | Function the task will run. |
    // | `"BTsender"` | Task name. |
    // | `6144` | Reserves 6,144 bytes of stack memory.. |
    // | First `nullptr` | No extra argument is passed to `bluetooth_sender`. |
    // | `1` | Task priority; higher numbers = higher priority. |
    // | Second `nullptr` | No not need FreeRTOS to return a task handle for later control. |
    // | `0` | Pin the task to ESP32 core 0. |

    if (xTaskCreatePinnedToCore(bluetooth_sender, "BTsender", 6144, nullptr, 1, nullptr, 0) != pdPASS) { // // pdPASS means task creation succeeded.
      is_bluetooth_ready = false; // Marks Bluetooth as unavailable and shuts it down, because there is no sender task to handle outgoing packets.
      SerialBT.end();
    }
  }

  memset(previous_lcd_rows, 0, sizeof(previous_lcd_rows)); // fills the entire previous_lcd_rows array with zeros. Before writing new text, your code compares it with this saved text.
  lcd.clear();
  update_lcd();
  // Start scheduling AFTER the startup delays.
  last_lcd_update_time_ms = millis();
  last_sample_time_us = micros();
}

void loop() {
  // All tasks below run once per 20 ms.
  uint32_t current_time_us = micros();
  uint32_t elapsed_time_us = current_time_us - last_sample_time_us;
  
  // Tests for time elapsed.
  // 1. Smaller than 20ms, do nothing, return and wait for next check.
  // 2. 40ms or more, restart to not break the filter.
  // 3. (else) Approx 20ms has gone by, so continue and execute all tasks.
  if (elapsed_time_us < SAMPLE_INTERVAL_US) return;
  if (elapsed_time_us >= 2 * SAMPLE_INTERVAL_US) {
    // A missed full tick breaks filter timing. Restart rather than
    // making up catch-up samples from repeated ADC reads.
    last_sample_time_us = current_time_us;
    is_filter_initialized = false;
    filter_warmup_sample_count = 0;
    threshold_window_sample_count = 0;
    is_signal_amplitude_sufficient = false;
    is_adaptive_threshold_ready = false;
    reset_beats();
    // Discard the partial packet; the skipped sequence number lets the
    // host see that time passed.
    samples_in_packet = 0;
    next_packet_sequence_number++;
    // Unobserved button time must not count towards a continuous hold.
    button_press_start_time_ms = millis();
  }
  else {
    last_sample_time_us += SAMPLE_INTERVAL_US;
  }
  uint32_t current_time_ms = millis();

  // REQUIREMENT #1 PROOF
  // Serial.println(current_time_ms);

  // 0. Pick up any alarm limits sent by the host.
  read_host_limits();

  // 1. Sample/debounce button and update blue recording LED.
  update_button(current_time_ms);
  digitalWrite(BLUE_LED_PIN, is_recording_active ? HIGH : LOW);

  // 2. Capture RAW PPG once, then filter a copy for detection.
  uint16_t raw_sample = analogRead(HR_SENSOR_PIN);
  float filtered_sample = band_pass_filter((float)raw_sample);
  // Collect 50 raw readings per packet (one second at 50 Hz).
  buffer_bluetooth_sample(raw_sample);

  // Gather samples for the threshold. True means the levels changed this cycle.
  bool thresholds_changed = collect_threshold_sample(filtered_sample);

  // Use the percentile amplitude check; clipping does not reset timing.
  // Saturated samples still appear unchanged in RawPPG for inspection.
  if (!is_signal_amplitude_sufficient) reset_beats();

  // 4. Find rising crossings and calculate BPM from beat intervals.
  detect_beat(filtered_sample, thresholds_changed, current_time_ms);
  if (has_previous_beat && current_time_ms - last_beat_time_ms > NO_BEAT_TIMEOUT_MS) {
    reset_beats();  // Do not keep displaying an old HR after pulse loss.
  }

  // Finish a beat flash after 100 ms, or immediately if HR becomes invalid.
  update_beat_led(current_time_ms);

  // 6. Sound an enabled alarm only for a valid out-of-range HR.
  update_alarm();

  // Send the same diagnostic fields to the USB Serial Plotter.
  print_debug_data(raw_sample, filtered_sample);

  // 8. Refresh LCD four times per second, without clearing/flickering.
  if (current_time_ms - last_lcd_update_time_ms >= LCD_MS) {
    last_lcd_update_time_ms = current_time_ms;
    update_lcd();
  }
}

// Each helper below performs one part of the 20 ms sampling cycle.
bool collect_threshold_sample(float filtered_sample) {
  // Do not collect threshold samples until the filter has settled.
  if (filter_warmup_sample_count < FILTER_WARMUP_SAMPLE_COUNT) {
    filter_warmup_sample_count++;
    return false;
  }

  threshold_sample_window[threshold_window_sample_count] = filtered_sample;
  threshold_window_sample_count++;

  if (threshold_window_sample_count < THRESHOLD_WINDOW_SAMPLE_COUNT) {
    return false;  // More samples are needed.
  }

  update_thresholds();
  threshold_window_sample_count = 0;
  return true;
}

void update_beat_led(uint32_t current_time_ms) {
  if (!is_signal_amplitude_sufficient || !is_heart_rate_valid ||
      (is_red_led_on && current_time_ms - red_led_flash_start_time_ms >= LED_FLASH_MS)) {
    is_red_led_on = false;
  }
  digitalWrite(RED_LED_PIN, is_red_led_on ? HIGH : LOW);
}

void print_debug_data(uint16_t raw_sample, float filtered_sample) {
  // Missing HR stays "---"; zero is only used as a placeholder in Bluetooth.
  if (is_heart_rate_valid) {
    Serial.printf(
      "RawPPG:%u\tFiltered:%.1f\tAdaptiveThreshold:%.1f\tHR:%.1f\t"
      "LowLimit:%.1f\tHighLimit:%.1f\n",
      raw_sample, filtered_sample, adaptive_threshold, heart_rate_bpm, low_hr_limit_bpm, high_hr_limit_bpm
    );
  }
  else {
   Serial.printf(
      "RawPPG:%u\tFiltered:%.1f\tAdaptiveThreshold:%.1f\tHR:---\t"
      "LowLimit:%.1f\tHighLimit:%.1f\n",
      raw_sample, filtered_sample, adaptive_threshold, low_hr_limit_bpm, high_hr_limit_bpm
    );
}
}

void update_button(uint32_t current_time_ms) {
  bool is_button_pressed = (digitalRead(BUTTON_PIN) == LOW);
  if (button.update(is_button_pressed)) {
    if (button.state()) {
      button_press_start_time_ms = current_time_ms;
      is_button_hold_action_triggered = false;
    }
    else if (!is_button_hold_action_triggered) {
      // Wait for release to distinguish a short press from a hold.
      is_recording_active = !is_recording_active;
    }
  }

  if (!is_button_pressed) button_press_start_time_ms = current_time_ms;  // Any sampled release breaks the hold.
  if (is_button_pressed && button.state() && !is_button_hold_action_triggered &&
      current_time_ms - button_press_start_time_ms >= BUTTON_MUTE_HOLD_MS) {
    is_button_hold_action_triggered = true;
    is_buzzer_enabled = !is_buzzer_enabled;
    update_lcd();  // Show the sound indicator immediately after the 3 s hold.
  }
}

float band_pass_filter(float raw_sample) {
  if (!is_filter_initialized) {
    // Avoid interpreting the initial sensor DC level as a heartbeat.
    previous_raw_sample = raw_sample;
    previous_high_pass_output = 0;
    previous_low_pass_output = 0;
    is_filter_initialized = true;
  }
  // First remove slow baseline changes, then reduce fast noise.
  float high_pass_output = high_pass_coefficient_b * (raw_sample - previous_raw_sample) - high_pass_coefficient_a * previous_high_pass_output;
  float low_pass_output = low_pass_coefficient_b * (high_pass_output + previous_high_pass_output) - low_pass_coefficient_a * previous_low_pass_output;
  // Save this sample's values so the next calculation can use them.
  previous_raw_sample = raw_sample;
  previous_high_pass_output = high_pass_output;
  previous_low_pass_output = low_pass_output;
  return low_pass_output;
}

int compare_floats(const void* a, const void* b) {
  float x = *(const float*)a;
  float y = *(const float*)b;
  // qsort needs a negative, positive, or zero result.
  if (x > y) return 1;
  if (x < y) return -1;
  return 0;
}

float percentile(float fraction) {
  // Interpolate between neighbouring values in the sorted window.
  float position = fraction * (THRESHOLD_WINDOW_SAMPLE_COUNT - 1);
  uint16_t index = (uint16_t)position;
  if (index >= THRESHOLD_WINDOW_SAMPLE_COUNT - 1) return sorted_threshold_sample_window[THRESHOLD_WINDOW_SAMPLE_COUNT - 1];
  return sorted_threshold_sample_window[index] + (position - index) *
         (sorted_threshold_sample_window[index + 1] - sorted_threshold_sample_window[index]);
}

void update_thresholds() {
  // Sort a copy: percentiles need ascending values, while the original
  // sample window keeps its order. P10/P90 reduce the influence of extremes.
  memcpy(sorted_threshold_sample_window, threshold_sample_window, sizeof(threshold_sample_window));
  qsort(sorted_threshold_sample_window, THRESHOLD_WINDOW_SAMPLE_COUNT, sizeof(float), compare_floats);
  float p10 = percentile(0.10f);
  float p90 = percentile(0.90f);
  float range = p90 - p10;
  adaptive_threshold = (p10 + p90) * 0.5f;

  // Use separate upper/lower levels. The signal must fall to the lower
  // level before another rise through the upper level can count as a beat.
  upper_beat_threshold = adaptive_threshold + THRESHOLD_HYSTERESIS_FRACTION * range;
  lower_beat_threshold = adaptive_threshold - THRESHOLD_HYSTERESIS_FRACTION * range;
  is_adaptive_threshold_ready = true;
  is_signal_amplitude_sufficient = (range >= MIN_VALID_FILTERED_RANGE);
  if (!is_signal_amplitude_sufficient) reset_beats();
}

void reset_beats() {
  has_previous_beat = false;
  heart_rate_bpm = 0;  // Internal placeholder only: is_heart_rate_valid=false means MISSING HR.
  is_heart_rate_valid = false;
  is_beat_detector_armed = false;
  is_red_led_on = false;
}

void detect_beat(float filtered_sample, bool thresholds_changed, uint32_t current_time_ms) {
  if (is_signal_amplitude_sufficient) {
    // Fall below the lower level before accepting an upward crossing.
    if (filtered_sample <= lower_beat_threshold) is_beat_detector_armed = true;
    if (!thresholds_changed && is_beat_detector_armed &&
        previous_filtered_sample < upper_beat_threshold && filtered_sample >= upper_beat_threshold) {
      is_beat_detector_armed = false;
      bool is_beat_accepted = false;
      if (!has_previous_beat) {
        last_beat_time_ms = current_time_ms;
        has_previous_beat = true;
        is_beat_accepted = true;  // First beat sets timing; LED waits for valid BPM.
      }
      else {
        uint32_t beat_interval_ms = current_time_ms - last_beat_time_ms;
        if (beat_interval_ms >= MIN_VALID_BEAT_INTERVAL_MS && beat_interval_ms <= MAX_VALID_BEAT_INTERVAL_MS) {
          heart_rate_bpm = 60000.0f / (float)beat_interval_ms;
          is_heart_rate_valid = true;
          last_beat_time_ms = current_time_ms;
          is_beat_accepted = true;
        }
        else {
          // Outside 40-150 BPM: hide HR and start a new reference.
          // Updating the reference for short intervals avoids combining
          // successive fast pulses into a falsely plausible slower rate.
          last_beat_time_ms = current_time_ms;
          heart_rate_bpm = 0;
          is_heart_rate_valid = false;
          is_red_led_on = false;
        }
      }
      if (is_beat_accepted) {
        is_red_led_on = true;
        red_led_flash_start_time_ms = current_time_ms;
      }
    }
  }
  // Skip threshold-update crossings so moving a level cannot make a beat.
  previous_filtered_sample = filtered_sample;
}

void read_host_limits() {
  // Bounded work per tick so the 20 ms sampling loop is never held up.
  // Obtain the values set by the host via BT
  for (int i = 0; i < 32 && is_bluetooth_ready && SerialBT.available(); i++) {
    char c = (char)SerialBT.read();
    if (c == '$') {
      host_alarm_limits_line_length = 0;  // a new message always starts here
    }
    if (c == '\n') {
      float low, high;
      if (parseLimits(host_alarm_limits_line, host_alarm_limits_line_length, &low, &high)) {
        low_hr_limit_bpm = low;
        high_hr_limit_bpm = high;
      }
      host_alarm_limits_line_length = 0;
    }
    else if (host_alarm_limits_line_length < sizeof(host_alarm_limits_line)) {
      host_alarm_limits_line[host_alarm_limits_line_length++] = c;
    }
    else {
      host_alarm_limits_line_length = 0;  // too long to be ours; wait for the next '$'
    }
  }
}

void update_alarm() {
  bool valid = is_signal_amplitude_sufficient && is_heart_rate_valid;
  bool is_outside_alarm_limits = heart_rate_bpm < low_hr_limit_bpm || heart_rate_bpm > high_hr_limit_bpm;
  // Normal HR and signal loss must not change the user's sound setting.
  bool sound = valid && is_outside_alarm_limits && is_buzzer_enabled;
  if (sound && !is_buzzer_currently_on) {
    tone(BUZZER_PIN, BUZZER_HZ);
    is_buzzer_currently_on = true;
  }
  else if (!sound && is_buzzer_currently_on) {
    noTone(BUZZER_PIN);
    digitalWrite(BUZZER_PIN, LOW);
    is_buzzer_currently_on = false;
  }
}

void write_lcd_row(uint8_t row, const char* text) {
  // Pad to 16 characters so shorter text overwrites the previous row.
  char padded[17];
  memset(padded, ' ', 16);
  padded[16] = '\0';
  size_t length = strlen(text);
  if (length > 16) length = 16;
  memcpy(padded, text, length);
  if (strcmp(padded, previous_lcd_rows[row]) != 0) {
    lcd.setCursor(0, row);
    lcd.print(padded);
    memcpy(previous_lcd_rows[row], padded, sizeof(padded));
  }
}

void update_lcd() {
  char row[17];  // 16 LCD characters, plus the string terminator.

  if (is_signal_amplitude_sufficient && is_heart_rate_valid) {
    const char* alarm_status = "";
    if (heart_rate_bpm < low_hr_limit_bpm) {
      alarm_status = "LOW";
    }
    else if (heart_rate_bpm > high_hr_limit_bpm) {
      alarm_status = "HIGH";
    }
    snprintf(row, sizeof(row), "HR:%5.1fBPM %s", heart_rate_bpm, alarm_status);
  }
  else if (is_adaptive_threshold_ready) {
    snprintf(row, sizeof(row), "HR: ---");
  }
  else {
    snprintf(row, sizeof(row), "HR: WAIT");
  }
  write_lcd_row(0, row);

  const char* bluetooth_status;
  if (!is_bluetooth_ready) {
    bluetooth_status = "Bluetooth: ERR";
  }
  else if (SerialBT.hasClient()) {
    bluetooth_status = "Bluetooth: YES";
  }
  else {
    bluetooth_status = "Bluetooth: NO";
  }

  // %-15s pads the status with spaces. The last column shows sound permission.
  char sound_indicator = ' ';
  if (!is_buzzer_enabled) {
    sound_indicator = '!';
  }
  snprintf(row, sizeof(row), "%-15s%c", bluetooth_status, sound_indicator);
  write_lcd_row(1, row);
}

void buffer_bluetooth_sample(uint16_t raw_sample) {
  // Store one reading at the next free array position, then increase the count.
  packet_being_filled.raw_ppg_samples[samples_in_packet] = raw_sample;
  samples_in_packet++;
  if (samples_in_packet < SAMPLES_PER_PACKET) return;

  // The packet is full: attach its number and the current measurement state.
  packet_being_filled.sequence_number = next_packet_sequence_number;
  next_packet_sequence_number++;
  packet_being_filled.hr_in_bpm = (is_signal_amplitude_sufficient && is_heart_rate_valid) ? heart_rate_bpm : 0.0f;
  packet_being_filled.recording = is_recording_active ? 1 : 0;
  // The queue copies the packet, so the capture buffer is reused at once.
  // Overwrite keeps only the newest packet if the sender falls behind.
  if (is_bluetooth_ready) xQueueOverwrite(bluetooth_packet_queue, &packet_being_filled);
  samples_in_packet = 0;
}

void bluetooth_sender(void* parameter) {
  BluetoothPacket packet;
  char frame[PACKET_BUF_SIZE];
  for (;;) {
    if (xQueueReceive(bluetooth_packet_queue, &packet, portMAX_DELAY) != pdTRUE) continue;
    if (!SerialBT.hasClient()) continue;

    size_t len = buildPacket(frame, sizeof(frame),
                             (uint16_t)(packet.sequence_number & 0xFFFF),
                             packet.hr_in_bpm, packet.recording != 0, packet.raw_ppg_samples);
    if (len != PACKET_LEN) continue;

    // REQUIREMENT #2 PROOF
    // static uint32_t previous_packet_ms = 0;
    // uint32_t now_ms = millis();
    // if (previous_packet_ms != 0)
    //   Serial.printf("Packet interval: %lu ms\n", (unsigned long)(now_ms - previous_packet_ms));
    // previous_packet_ms = now_ms;

    // A slow write waits here, outside the sampling loop.
    size_t sent = SerialBT.write((const uint8_t*)frame, PACKET_LEN);
    if (sent != PACKET_LEN && SerialBT.hasClient()) {
      // End a partial line so the host rejects it and resynchronises.
      SerialBT.write((uint8_t)'\n');
    }
  }
}

void startup_sequence() {
  // Show the name immediately on the bottom row, one column left.
  lcd.setCursor(3, 1);
  lcd.print("LumiPulse");

  // Keep both LEDs steadily ON for 3 seconds, starting with the tune.
  // These are the only delay() calls: sampling has not started yet.
  digitalWrite(RED_LED_PIN, HIGH);
  digitalWrite(BLUE_LED_PIN, HIGH);
  const uint16_t notes[5] = {1047, 1319, 1568, 2093, 2637};
  for (uint8_t i = 0; i < 5; i++) {
    if (is_buzzer_enabled) tone(BUZZER_PIN, notes[i]);  // Silent while muted.
    delay(100);
    noTone(BUZZER_PIN);
    delay(100);
  }
  delay(2000);  // 1 second of music + 2 seconds = 3 seconds total.
  digitalWrite(RED_LED_PIN, LOW);
  digitalWrite(BLUE_LED_PIN, LOW);
}

