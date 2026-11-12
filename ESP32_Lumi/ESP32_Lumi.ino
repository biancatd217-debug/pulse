/*
** BMET2922 - LumiPulse (local monitor + Bluetooth SPP link to the host GUI)
**
** Authors: Abigail Cameron (MCU: sensing, filtering, beat detection, button,
**          LEDs, LCD, buzzer, alarm)
**         
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
#include <Arduino.h>
#include <LiquidCrystal.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "BluetoothSerial.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "switch.h"   // provided button debounce
#include "packet.h"   // Bluetooth packet format

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

LiquidCrystal lcd(LCD_RS_PIN, LCD_ENABLE_PIN, LCD_D4_PIN, LCD_D5_PIN, LCD_D6_PIN, LCD_D7_PIN);
Switch button(BUTTON_PIN, DEBOUNCE_COUNT);  // provided debounce library

const uint32_t BUTTON_MUTE_HOLD_MS = 3000UL;
const uint32_t LED_FLASH_MS = 100UL;
const uint32_t LCD_MS = 250UL;
const uint16_t BUZZER_HZ = 2000;
constexpr uint32_t SAMPLE_INTERVAL_US = 20000UL;  // 20 ms sampling (50 Hz)
constexpr float SAMPLE_HZ = 1000000.0f / SAMPLE_INTERVAL_US;

// Filter cutoffs (0.5 Hz = 30 BPM, 5 Hz = 300 BPM)
const float HIGH_PASS_CUTOFF_HZ = 0.5f;  // removes baseline drift
const float LOW_PASS_CUTOFF_HZ = 5.0f;   // removes fast noise

// Heartrate values that trigger alarm
const float LOW_HR_ALARM_THRESHOLD_BPM = 50.0f;
const float HIGH_HR_ALARM_THRESHOLD_BPM = 100.0f;

// Valid beat interval: 400-1500 ms = 40-150 BPM. Independent of the alarm limits.
const uint32_t MIN_VALID_BEAT_INTERVAL_MS = 400UL;
const uint32_t MAX_VALID_BEAT_INTERVAL_MS = 1500UL;

// Two seconds, to settle the filter, then two-second percentile windows.
constexpr uint16_t FILTER_WARMUP_SAMPLE_COUNT = static_cast<uint16_t>(SAMPLE_HZ * 2.0f);
constexpr uint16_t THRESHOLD_WINDOW_SAMPLE_COUNT = static_cast<uint16_t>(SAMPLE_HZ * 2.0f);

// Provisional amplitude check; calibrate with real recordings.
const float MIN_VALID_FILTERED_RANGE = 100.0f;
const float THRESHOLD_HYSTERESIS_FRACTION = 0.05f;
bool is_signal_amplitude_sufficient = false;

// Clear HR after 1.7 s without an accepted beat.
const uint32_t NO_BEAT_TIMEOUT_MS = MAX_VALID_BEAT_INTERVAL_MS + 200UL;

// ---------------- BLUETOOTH SETTINGS (Bianca Dibden) ----------------
BluetoothSerial SerialBT;
const char* DEVICE_NAME = "AbiESP32";

bool is_bluetooth_ready = false;

// One packet held in memory before it is formatted as ASCII (see packet.h).
struct BluetoothPacket {
  uint32_t sequence_number;  // wraps to 16 bits on the wire; host detects gaps
  float hr_in_bpm;           // 0 = no valid measurement
  uint8_t recording;         // 0 = off, 1 = on
  uint16_t raw_ppg_samples[SAMPLES_PER_PACKET];
};

// Sampling code fills packet_being_filled and hands complete packets to the
// Bluetooth task through this queue, so a slow write never delays sampling.
QueueHandle_t bluetooth_packet_queue = nullptr;
BluetoothPacket packet_being_filled = {};

uint16_t samples_in_packet = 0;
uint32_t next_packet_sequence_number = 0;

// ---------------- ALARM LIMITS ----------------
// Defaults, replaced by limits sent from the host (Bianca Dibden's read_host_limits).
float low_hr_limit_bpm = LOW_HR_ALARM_THRESHOLD_BPM;
float high_hr_limit_bpm = HIGH_HR_ALARM_THRESHOLD_BPM;
char host_alarm_limits_line[24] = {};  // "$LIM,050.0,100.0*CC" is 19 chars
uint8_t host_alarm_limits_line_length = 0;

// ---------------- SCHEDULER AND BUTTON SETTINGS ----------------
uint32_t last_sample_time_us = 0;
uint32_t last_lcd_update_time_ms = 0;
bool is_recording_active = false;
bool is_button_hold_action_triggered = false;  // one long press triggers only once
uint32_t button_press_start_time_ms = 0;
bool is_buzzer_enabled = true;
bool is_buzzer_currently_on = false;

// ---------------- FILTER AND THRESHOLD SETTINGS ----------------
// Pipeline: raw -> high-pass -> low-pass -> adaptive threshold -> beat interval -> HR.
// Coefficients are calculated in setup().
float high_pass_coefficient_b = 0.0f;
float high_pass_coefficient_a = 0.0f;
float low_pass_coefficient_b = 0.0f;
float low_pass_coefficient_a = 0.0f;

// Filter state
float previous_raw_sample = 0.0f;
float previous_high_pass_output = 0.0f;
float previous_low_pass_output = 0.0f;
bool is_filter_initialized = false;

uint16_t filter_warmup_sample_count = 0;
float threshold_sample_window[THRESHOLD_WINDOW_SAMPLE_COUNT] = {};
float sorted_threshold_sample_window[THRESHOLD_WINDOW_SAMPLE_COUNT] = {};  // for P10/P90
uint16_t threshold_window_sample_count = 0;

// The first threshold is ready after about 4 s (2 s warm-up + 2 s window).
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
// The host packet carries 0.0 when this is false.
bool is_heart_rate_valid = false;
bool is_red_led_on = false;
uint32_t red_led_flash_start_time_ms = 0;
char previous_lcd_rows[2][17] = {{0}, {0}};

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

// Author: Abigail Cameron (Bluetooth start-up block: Bianca Dibden)
void setup() {
  pinMode(RED_LED_PIN, OUTPUT);
  pinMode(BLUE_LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(HR_SENSOR_PIN, INPUT);
  digitalWrite(RED_LED_PIN, LOW);
  digitalWrite(BLUE_LED_PIN, LOW);
  digitalWrite(BUZZER_PIN, LOW);
  Serial.begin(115200);
  analogReadResolution(12);  // 0-4095 ADC counts
  analogSetPinAttenuation(HR_SENSOR_PIN, ADC_11db);

  // First-order high-pass + low-pass (nominal cutoffs)
  float k = tanf(PI * HIGH_PASS_CUTOFF_HZ / SAMPLE_HZ);
  high_pass_coefficient_b = 1.0f / (1.0f + k);
  high_pass_coefficient_a = (k - 1.0f) / (k + 1.0f);
  k = tanf(PI * LOW_PASS_CUTOFF_HZ / SAMPLE_HZ);
  low_pass_coefficient_b = k / (1.0f + k);
  low_pass_coefficient_a = (k - 1.0f) / (k + 1.0f);

  lcd.begin(16, 2);
  startup_sequence();

  // Bluetooth (Bianca Dibden): start once per boot, after the startup delays.
  bluetooth_packet_queue = xQueueCreate(1, sizeof(BluetoothPacket));
  if (bluetooth_packet_queue != nullptr && SerialBT.begin(DEVICE_NAME)) {
    is_bluetooth_ready = true;
    // Sender runs as its own task (core 0) so Bluetooth writes cannot delay sampling.
    if (xTaskCreatePinnedToCore(bluetooth_sender, "BTsender", 6144, nullptr, 1, nullptr, 0) != pdPASS) {
      is_bluetooth_ready = false;
      SerialBT.end();
    }
  }

  memset(previous_lcd_rows, 0, sizeof(previous_lcd_rows));
  lcd.clear();
  update_lcd();
  last_lcd_update_time_ms = millis();
  last_sample_time_us = micros();
}

// Author: Abigail Cameron (read_host_limits and buffer_bluetooth_sample calls: Bianca Dibden)
void loop() {
  uint32_t current_time_us = micros();
  uint32_t elapsed_time_us = current_time_us - last_sample_time_us;
  
  // Wait for the next 20 ms tick; after a missed tick (>= 40 ms) restart.
  if (elapsed_time_us < SAMPLE_INTERVAL_US) return;
  if (elapsed_time_us >= 2 * SAMPLE_INTERVAL_US) {
    last_sample_time_us = current_time_us;
    is_filter_initialized = false;
    filter_warmup_sample_count = 0;
    threshold_window_sample_count = 0;
    is_signal_amplitude_sufficient = false;
    is_adaptive_threshold_ready = false;
    reset_beats();
    // Discard the partial packet; the skipped sequence number shows the gap to the host.
    samples_in_packet = 0;
    next_packet_sequence_number++;
    button_press_start_time_ms = millis();
  }
  else {
    last_sample_time_us += SAMPLE_INTERVAL_US;
  }
  uint32_t current_time_ms = millis();

  // Pick up any alarm limits sent by the host
  read_host_limits();

  // Button and recording LED
  update_button(current_time_ms);
  digitalWrite(BLUE_LED_PIN, is_recording_active ? HIGH : LOW);

  // Sample, filter, buffer raw data for the packet
  uint16_t raw_sample = analogRead(HR_SENSOR_PIN);
  float filtered_sample = band_pass_filter((float)raw_sample);
  buffer_bluetooth_sample(raw_sample);

  bool thresholds_changed = collect_threshold_sample(filtered_sample);

  if (!is_signal_amplitude_sufficient) reset_beats();

  // Beat detection
  detect_beat(filtered_sample, thresholds_changed, current_time_ms);
  if (has_previous_beat && current_time_ms - last_beat_time_ms > NO_BEAT_TIMEOUT_MS) {
    reset_beats();  // pulse lost
  }

  update_beat_led(current_time_ms);

  update_alarm();

  print_debug_data(raw_sample, filtered_sample);

  if (current_time_ms - last_lcd_update_time_ms >= LCD_MS) {
    last_lcd_update_time_ms = current_time_ms;
    update_lcd();
  }
}

// Author: Abigail Cameron
bool collect_threshold_sample(float filtered_sample) {
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

// Author: Abigail Cameron
void update_beat_led(uint32_t current_time_ms) {
  if (!is_signal_amplitude_sufficient || !is_heart_rate_valid ||
      (is_red_led_on && current_time_ms - red_led_flash_start_time_ms >= LED_FLASH_MS)) {
    is_red_led_on = false;
  }
  digitalWrite(RED_LED_PIN, is_red_led_on ? HIGH : LOW);
}

// Author: Abigail Cameron
void print_debug_data(uint16_t raw_sample, float filtered_sample) {
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

// Author: Abigail Cameron
void update_button(uint32_t current_time_ms) {
  bool is_button_pressed = (digitalRead(BUTTON_PIN) == LOW);
  if (button.update(is_button_pressed)) {
    if (button.state()) {
      button_press_start_time_ms = current_time_ms;
      is_button_hold_action_triggered = false;
    }
    else if (!is_button_hold_action_triggered) {
          is_recording_active = !is_recording_active;
    }
  }

  if (!is_button_pressed) button_press_start_time_ms = current_time_ms;
  if (is_button_pressed && button.state() && !is_button_hold_action_triggered &&
      current_time_ms - button_press_start_time_ms >= BUTTON_MUTE_HOLD_MS) {
    is_button_hold_action_triggered = true;
    is_buzzer_enabled = !is_buzzer_enabled;
    update_lcd();  // Show the sound indicator immediately after the 3 s hold.
  }
}

// Author: Abigail Cameron
float band_pass_filter(float raw_sample) {
  if (!is_filter_initialized) {
    previous_raw_sample = raw_sample;
    previous_high_pass_output = 0;
    previous_low_pass_output = 0;
    is_filter_initialized = true;
  }
  float high_pass_output = high_pass_coefficient_b * (raw_sample - previous_raw_sample) - high_pass_coefficient_a * previous_high_pass_output;
  float low_pass_output = low_pass_coefficient_b * (high_pass_output + previous_high_pass_output) - low_pass_coefficient_a * previous_low_pass_output;
  previous_raw_sample = raw_sample;
  previous_high_pass_output = high_pass_output;
  previous_low_pass_output = low_pass_output;
  return low_pass_output;
}

// Author: Abigail Cameron
int compare_floats(const void* a, const void* b) {
  float x = *(const float*)a;
  float y = *(const float*)b;
  if (x > y) return 1;
  if (x < y) return -1;
  return 0;
}

// Author: Abigail Cameron
float percentile(float fraction) {
  float position = fraction * (THRESHOLD_WINDOW_SAMPLE_COUNT - 1);
  uint16_t index = (uint16_t)position;
  if (index >= THRESHOLD_WINDOW_SAMPLE_COUNT - 1) return sorted_threshold_sample_window[THRESHOLD_WINDOW_SAMPLE_COUNT - 1];
  return sorted_threshold_sample_window[index] + (position - index) *
         (sorted_threshold_sample_window[index + 1] - sorted_threshold_sample_window[index]);
}

// Author: Abigail Cameron
void update_thresholds() {
  // P10/P90 of a sorted copy reduce the influence of extremes
  memcpy(sorted_threshold_sample_window, threshold_sample_window, sizeof(threshold_sample_window));
  qsort(sorted_threshold_sample_window, THRESHOLD_WINDOW_SAMPLE_COUNT, sizeof(float), compare_floats);
  float p10 = percentile(0.10f);
  float p90 = percentile(0.90f);
  float range = p90 - p10;
  adaptive_threshold = (p10 + p90) * 0.5f;

  // Hysteresis: must fall below the lower level before the next beat counts
  upper_beat_threshold = adaptive_threshold + THRESHOLD_HYSTERESIS_FRACTION * range;
  lower_beat_threshold = adaptive_threshold - THRESHOLD_HYSTERESIS_FRACTION * range;
  is_adaptive_threshold_ready = true;
  is_signal_amplitude_sufficient = (range >= MIN_VALID_FILTERED_RANGE);
  if (!is_signal_amplitude_sufficient) reset_beats();
}

// Author: Abigail Cameron
void reset_beats() {
  has_previous_beat = false;
  heart_rate_bpm = 0;  // Internal placeholder only: is_heart_rate_valid=false means MISSING HR.
  is_heart_rate_valid = false;
  is_beat_detector_armed = false;
  is_red_led_on = false;
}

// Author: Abigail Cameron
void detect_beat(float filtered_sample, bool thresholds_changed, uint32_t current_time_ms) {
  if (is_signal_amplitude_sufficient) {
    if (filtered_sample <= lower_beat_threshold) is_beat_detector_armed = true;
    if (!thresholds_changed && is_beat_detector_armed &&
        previous_filtered_sample < upper_beat_threshold && filtered_sample >= upper_beat_threshold) {
      is_beat_detector_armed = false;
      bool is_beat_accepted = false;
      if (!has_previous_beat) {
        last_beat_time_ms = current_time_ms;
        has_previous_beat = true;
        is_beat_accepted = true;  // first beat only sets timing
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
          // Outside 40-150 BPM: hide HR and restart the reference
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
  previous_filtered_sample = filtered_sample;
}

// Author: Bianca Dibden
void read_host_limits() {
  // Bounded work per tick so sampling is never held up
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

// Author: Abigail Cameron
void update_alarm() {
  bool valid = is_signal_amplitude_sufficient && is_heart_rate_valid;
  bool is_outside_alarm_limits = heart_rate_bpm < low_hr_limit_bpm || heart_rate_bpm > high_hr_limit_bpm;
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

// Author: Abigail Cameron
void write_lcd_row(uint8_t row, const char* text) {
  // Pad to 16 chars; only redraw if the row changed
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

// Author: Abigail Cameron
void update_lcd() {
  char row[17];

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

  // Last column shows '!' when the buzzer is disabled
  char sound_indicator = ' ';
  if (!is_buzzer_enabled) {
    sound_indicator = '!';
  }
  snprintf(row, sizeof(row), "%-15s%c", bluetooth_status, sound_indicator);
  write_lcd_row(1, row);
}

// Author: Bianca Dibden
void buffer_bluetooth_sample(uint16_t raw_sample) {
  packet_being_filled.raw_ppg_samples[samples_in_packet] = raw_sample;
  samples_in_packet++;
  if (samples_in_packet < SAMPLES_PER_PACKET) return;

  packet_being_filled.sequence_number = next_packet_sequence_number;
  next_packet_sequence_number++;
  packet_being_filled.hr_in_bpm = (is_signal_amplitude_sufficient && is_heart_rate_valid) ? heart_rate_bpm : 0.0f;
  packet_being_filled.recording = is_recording_active ? 1 : 0;
  // Overwrite keeps only the newest packet if the sender falls behind
  if (is_bluetooth_ready) xQueueOverwrite(bluetooth_packet_queue, &packet_being_filled);
  samples_in_packet = 0;
}

// Author: Bianca Dibden
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

    size_t sent = SerialBT.write((const uint8_t*)frame, PACKET_LEN);
    if (sent != PACKET_LEN && SerialBT.hasClient()) {
      // End a partial line so the host rejects it and resyncs
      SerialBT.write((uint8_t)'\n');
    }
  }
}

// Author: Abigail Cameron
void startup_sequence() {
  lcd.setCursor(3, 1);
  lcd.print("LumiPulse");

  // Only delay() calls in the sketch: sampling has not started yet
  digitalWrite(RED_LED_PIN, HIGH);
  digitalWrite(BLUE_LED_PIN, HIGH);
  const uint16_t notes[5] = {1047, 1319, 1568, 2093, 2637};
  for (uint8_t i = 0; i < 5; i++) {
    tone(BUZZER_PIN, notes[i]);
    delay(100);
    noTone(BUZZER_PIN);
    delay(100);
  }
  delay(2000);
  digitalWrite(RED_LED_PIN, LOW);
  digitalWrite(BLUE_LED_PIN, LOW);
}

