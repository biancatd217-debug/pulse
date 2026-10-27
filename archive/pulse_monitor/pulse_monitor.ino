/*
** BMET2922 - Pulse Monitoring System (LumiPulse) - integrated ESP32 sketch
**
** PPG and button sampled every 20 ms.
** Red LED: flashes on accepted beats.
** Blue LED: ON when recording is enabled.
** Button: active LOW, debounce count 5.
** Buzzer: sounds below 40 or above 200 BPM.
** LCD: displays live HR and recording/alarm state.
** Bluetooth SPP: every 50 samples (1 s) a 268-byte packet is sent to the
**   host with sequence number, MCU HR, recording state and the 50 raw
**   samples (format in packet.h; host side in Code/host/comms.py).
**
** Acquisition, MCU HR, button, LEDs, buzzer, LCD, Serial Plotter: Abi
**   (original sketch: 2ledsbuzzerscreen.ino, unchanged logic)
** Bluetooth link, packet construction, replay test mode: Bianca Dibden
**   (sections marked "BLUETOOTH")
*/

#include "switch.h"
#include <LiquidCrystal.h>
#include <string.h>

// BLUETOOTH (Bianca)
#include "BluetoothSerial.h"
#include "packet.h"
#include "pulse_data.h"

#if !defined(CONFIG_BT_ENABLED) || !defined(CONFIG_BLUEDROID_ENABLED)
#error Bluetooth is not enabled! Select a classic ESP32 board (ESP32 Dev Module).
#endif

// ======================================================
// PINS
// ======================================================

#define RED_LED_PIN 32
#define BLUE_LED_PIN 26
#define SWITCH_PIN 33
#define SENSOR_PIN 34
#define BUZZER_PIN 27

// LCD connections: RS, E, D4, D5, D6, D7
LiquidCrystal lcd(23, 22, 21, 19, 18, 25);

// ======================================================
// BLUETOOTH CONFIGURATION (Bianca)
// ======================================================

// 1 = read the PPG sensor; 0 = replay pulse_data.h (course pulse.csv)
// so the whole system can be tested without the sensor attached.
#define USE_REAL_SENSOR 1

// 1 = also echo each packet to the USB Serial Monitor (debug only).
// Keep 0 normally: packet text would corrupt the Serial Plotter (Reqt 4).
#define PRINT_PACKETS_TO_USB 0

const char *BT_NAME = "LumiPulse-G01";

// ======================================================
// LCD
// ======================================================

#define LCD_UPDATE_MS 250UL

unsigned long lastLcdUpdate = 0;
char previousLcdRows[2][17] = {{0}, {0}};

// ======================================================
// BUTTON AND RECORDING
// ======================================================

#define DEBOUNCE_CNT 5

Switch switch1(1, DEBOUNCE_CNT);

// Recording state: sent to the host in every Bluetooth packet; the host
// GUI starts/stops CSV recording from it.
bool recording = false;

// ======================================================
// PPG AND ADAPTIVE THRESHOLD
// ======================================================

uint16_t sensor_reading = 0;
uint16_t adaptive_threshold = 1800;

uint16_t signal_min = 4095;
uint16_t signal_max = 0;
uint16_t threshold_sample_count = 0;

// 100 samples x 20 ms = 2 seconds
#define THRESHOLD_WINDOW_SAMPLES 100
#define MIN_SIGNAL_RANGE 300

// Basic amplitude check, not complete signal validation.
bool signal_valid = false;

bool old_PT_state = false;
bool current_PT_state = false;

// ======================================================
// HEART RATE
// ======================================================

// Detection range: 20 to 240 BPM.
// Wider than the alarm limits so alarms can be detected.
#define MIN_PULSE_PERIOD 250UL
#define MAX_PULSE_PERIOD 3000UL
#define NO_PULSE_TIMEOUT 3500UL

unsigned long last_transition = 0;
bool have_previous_beat = false;

float heart_rate = 0.0;
bool heart_rate_valid = false;

// ======================================================
// RED LED
// ======================================================

#define BEAT_FLASH_MS 100UL

bool beat_led_on = false;
unsigned long beat_flash_start = 0;

// ======================================================
// BUZZER
// ======================================================

#define LOW_HR_LIMIT 40.0f
#define HIGH_HR_LIMIT 200.0f
#define ALARM_FREQUENCY 2000

bool buzzer_on = false;

// ======================================================
// SCHEDULER
// ======================================================

#define TICK_20MSEC 20000UL

unsigned long lastTickTime = 0;

// ======================================================
// BLUETOOTH STATE (Bianca)
// ======================================================

BluetoothSerial SerialBT;

uint16_t packet_samples[SAMPLES_PER_PACKET];   // last 1 s of raw samples
uint8_t packet_sample_count = 0;
uint16_t packet_seq = 0;                        // wraps 65535 -> 0
char packet_buffer[PACKET_BUF_SIZE];

// Set from the Bluetooth SPP open/close events.
// TODO (team decision pending): show the connection status to the user
// (e.g. LCD indicator or blue LED pattern). Not displayed yet.
volatile bool bt_connected = false;

unsigned long packets_sent = 0;
unsigned long packets_not_sent = 0;
uint16_t replay_index = 0;

void btCallback(esp_spp_cb_event_t event, esp_spp_cb_param_t *param);
uint16_t readSample();
void sendPacket();

// ======================================================
// SETUP
// ======================================================

void setup() {
  pinMode(RED_LED_PIN, OUTPUT);
  pinMode(BLUE_LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(SENSOR_PIN, INPUT);

  // Button connects GPIO33 to GND.
  pinMode(SWITCH_PIN, INPUT_PULLUP);

  digitalWrite(RED_LED_PIN, LOW);
  digitalWrite(BLUE_LED_PIN, LOW);
  digitalWrite(BUZZER_PIN, LOW);

  Serial.begin(115200);
  analogReadResolution(12);

  lcd.begin(16, 2);
  updateLcd();

  // BLUETOOTH (Bianca): start the SPP server the host connects to.
  SerialBT.register_callback(btCallback);
  SerialBT.begin(BT_NAME);

  lastLcdUpdate = millis();
  lastTickTime = micros();
}

// ======================================================
// MAIN LOOP
// ======================================================

void loop() {
  if ((micros() - lastTickTime) < TICK_20MSEC) {
    return;
  }

  lastTickTime = micros();
  unsigned long now = millis();

  // ----------------------------------------------------
  // 1. BUTTON AND BLUE LED
  // ----------------------------------------------------

  bool button_pressed = (digitalRead(SWITCH_PIN) == LOW);

  if (switch1.update(button_pressed)) {
    if (switch1.state()) {
      recording = !recording;
    }
  }

  digitalWrite(BLUE_LED_PIN, recording ? HIGH : LOW);

  // ----------------------------------------------------
  // 2. READ PPG AND TRACK SIGNAL RANGE
  // ----------------------------------------------------

  sensor_reading = readSample();

  // BLUETOOTH (Bianca): keep the last second of samples for the packet.
  packet_samples[packet_sample_count++] = sensor_reading;

  if (sensor_reading < signal_min) {
    signal_min = sensor_reading;
  }

  if (sensor_reading > signal_max) {
    signal_max = sensor_reading;
  }

  threshold_sample_count++;

  // ----------------------------------------------------
  // 3. ADAPTIVE THRESHOLD
  // ----------------------------------------------------

  if (threshold_sample_count >= THRESHOLD_WINDOW_SAMPLES) {
    uint16_t signal_range = signal_max - signal_min;

    signal_valid = (signal_range >= MIN_SIGNAL_RANGE);

    if (signal_valid) {
      adaptive_threshold = (signal_min + signal_max) / 2;
    }
    else {
      heart_rate = 0.0;
      heart_rate_valid = false;
      have_previous_beat = false;

      old_PT_state = false;
      current_PT_state = false;
      beat_led_on = false;
    }

    signal_min = 4095;
    signal_max = 0;
    threshold_sample_count = 0;
  }

  // ----------------------------------------------------
  // 4. CREATE PULSE TRAIN
  // ----------------------------------------------------

  old_PT_state = current_PT_state;

  current_PT_state =
      signal_valid && (sensor_reading > adaptive_threshold);

  // ----------------------------------------------------
  // 5. DETECT BEATS AND CALCULATE HR
  // ----------------------------------------------------

  if (signal_valid && !old_PT_state && current_PT_state) {
    if (!have_previous_beat) {
      // First beat establishes a timing reference.
      last_transition = now;
      have_previous_beat = true;
    }
    else {
      unsigned long candidate_period = now - last_transition;

      if (candidate_period >= MIN_PULSE_PERIOD &&
          candidate_period <= MAX_PULSE_PERIOD) {

        heart_rate = 60000.0f / (float)candidate_period;
        heart_rate_valid = true;
        last_transition = now;

        beat_led_on = true;
        beat_flash_start = now;
      }
      else if (candidate_period > MAX_PULSE_PERIOD) {
        // Start a fresh reference after an excessive interval.
        last_transition = now;
        heart_rate = 0.0;
        heart_rate_valid = false;
        beat_led_on = false;
      }

      // Ignore edges that arrive too soon.
    }
  }

  // ----------------------------------------------------
  // 6. EXPIRE STALE HR
  // ----------------------------------------------------

  if (have_previous_beat &&
      (now - last_transition > MAX_PULSE_PERIOD)) {

    heart_rate = 0.0;
    heart_rate_valid = false;
    beat_led_on = false;
  }

  if (have_previous_beat &&
      (now - last_transition > NO_PULSE_TIMEOUT)) {

    have_previous_beat = false;
  }

  // ----------------------------------------------------
  // 7. RED LED FLASH
  // ----------------------------------------------------

  if (!signal_valid ||
      (beat_led_on &&
       (now - beat_flash_start >= BEAT_FLASH_MS))) {

    beat_led_on = false;
  }

  digitalWrite(RED_LED_PIN, beat_led_on ? HIGH : LOW);

  // ----------------------------------------------------
  // 8. HIGH / LOW HR BUZZER
  // ----------------------------------------------------

  bool alarm_required =
      signal_valid &&
      heart_rate_valid &&
      (heart_rate < LOW_HR_LIMIT ||
       heart_rate > HIGH_HR_LIMIT);

  if (alarm_required && !buzzer_on) {
    tone(BUZZER_PIN, ALARM_FREQUENCY);
    buzzer_on = true;
  }
  else if (!alarm_required && buzzer_on) {
    noTone(BUZZER_PIN);
    digitalWrite(BUZZER_PIN, LOW);
    buzzer_on = false;
  }

  // ----------------------------------------------------
  // 9. SERIAL PLOTTER
  // ----------------------------------------------------

  Serial.printf(
    "RawPPG:%u\tAdaptiveThreshold:%u\tHeartRate:%.1f\n",
    sensor_reading,
    adaptive_threshold,
    heart_rate
  );

  // ----------------------------------------------------
  // 10. BLUETOOTH PACKET EVERY 50 SAMPLES = 1 s (Bianca)
  // ----------------------------------------------------

  if (packet_sample_count >= SAMPLES_PER_PACKET) {
    sendPacket();
    packet_sample_count = 0;
  }

  // ----------------------------------------------------
  // 11. LCD REFRESH
  // ----------------------------------------------------

  if (now - lastLcdUpdate >= LCD_UPDATE_MS) {
    lastLcdUpdate = now;
    updateLcd();
  }
}

// ======================================================
// LCD DISPLAY
// ======================================================

void updateLcd() {
  char hrText[17];
  char statusText[17];

  bool displayValid = signal_valid && heart_rate_valid;

  if (displayValid) {
    snprintf(
      hrText,
      sizeof(hrText),
      "HR: %.1f BPM",
      heart_rate
    );
  }
  else {
    snprintf(hrText, sizeof(hrText), "HR: -- BPM");
  }

  const char* pulseStatus = "NO PULSE";

  if (displayValid) {
    if (heart_rate < LOW_HR_LIMIT) {
      pulseStatus = "LOW";
    }
    else if (heart_rate > HIGH_HR_LIMIT) {
      pulseStatus = "HIGH";
    }
    else {
      pulseStatus = "OK";
    }
  }

  snprintf(
    statusText,
    sizeof(statusText),
    "REC:%s %s",
    recording ? "ON" : "OFF",
    pulseStatus
  );

  writeLcdRow(0, hrText);
  writeLcdRow(1, statusText);
}

// ======================================================
// WRITE A PADDED LCD ROW WITHOUT CLEARING THE SCREEN
// ======================================================

void writeLcdRow(uint8_t row, const char* text) {
  char padded[17];

  memset(padded, ' ', 16);
  padded[16] = '\0';

  size_t length = strlen(text);

  if (length > 16) {
    length = 16;
  }

  memcpy(padded, text, length);

  // Rewrite only when content changes.
  if (strcmp(padded, previousLcdRows[row]) != 0) {
    lcd.setCursor(0, row);
    lcd.print(padded);

    memcpy(previousLcdRows[row], padded, 17);
  }
}

// ======================================================
// BLUETOOTH (Bianca)
// ======================================================

void btCallback(esp_spp_cb_event_t event, esp_spp_cb_param_t *param)
/*
** Called by the Bluetooth stack on SPP events. Kept minimal because it runs
** in the Bluetooth task: it only records whether a host is connected.
**
** Author: Bianca Dibden
*/
{
  (void)param;
  if (event == ESP_SPP_SRV_OPEN_EVT) {
    bt_connected = true;
  } else if (event == ESP_SPP_CLOSE_EVT) {
    bt_connected = false;
  }
}

uint16_t readSample()
/*
** Returns the next PPG sample (12-bit ADC count). With USE_REAL_SENSOR 0
** it replays the course recording in pulse_data.h (looping), so Abi's
** MCU HR and the host pipeline can be tested without the sensor.
**
** Author: Bianca Dibden
*/
{
#if USE_REAL_SENSOR
  return analogRead(SENSOR_PIN);
#else
  uint16_t v = PULSE_DATA[replay_index];
  replay_index = (replay_index + 1) % PULSE_DATA_LEN;
  return v;
#endif
}

void sendPacket()
/*
** Builds the packet from the last 50 samples and sends it over Bluetooth
** (Reqt 18). MCU HR is sent as 0.0 when Abi's algorithm has no valid
** pulse. If no host is connected the packet is not sent, but the sequence
** number still advances so the host can see how much time passed.
**
** Author: Bianca Dibden
*/
{
  float mcu_hr = (signal_valid && heart_rate_valid) ? heart_rate : 0.0f;

  size_t len = buildPacket(packet_buffer, sizeof(packet_buffer), packet_seq,
                           mcu_hr, recording, packet_samples);

  if (len == PACKET_LEN && bt_connected) {
    SerialBT.write((const uint8_t *)packet_buffer, len);
    packets_sent++;
  } else {
    packets_not_sent++;
  }

#if PRINT_PACKETS_TO_USB
  Serial.write((const uint8_t *)packet_buffer, len);
#endif

  packet_seq++;
}
