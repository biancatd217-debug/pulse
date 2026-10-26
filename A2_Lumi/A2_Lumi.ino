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
** Bluetooth: device name BT_NAME. Every 50 samples (1 s) one fixed-length
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

#include <Arduino.h>
#include <LiquidCrystal.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "BluetoothSerial.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "switch.h"
#include "packet.h"

#if !defined(CONFIG_BT_SPP_ENABLED)
#error This sketch requires a classic ESP32 with Bluetooth SPP enabled.
#endif

// ---------------- PINS AND SETTINGS ----------------
#define RED_LED_PIN 17
#define BLUE_LED_PIN 26
#define SWITCH_PIN 33
#define SENSOR_PIN 34
#define BUZZER_PIN 27
#define DEBOUNCE_CNT 5

LiquidCrystal lcd(23, 22, 21, 19, 18, 25);
Switch switch1(1, DEBOUNCE_CNT);

const uint32_t TICK_20MSEC = 20000UL;
const float SAMPLE_HZ = 50.0f;
const float HP_HZ = 0.5f;
const float LP_HZ = 5.0f;
const float DEFAULT_LOW_HR_LIMIT = 50.0f;
const float DEFAULT_HIGH_HR_LIMIT = 100.0f;
const uint32_t HOLD_MS = 3000UL;
const uint32_t FLASH_MS = 100UL;
const uint32_t LCD_MS = 250UL;
const uint16_t BUZZER_HZ = 2000;

// Measurement limits: 40-150 BPM. Alarm limits above are independent.
const uint32_t MIN_BEAT_MS = 400UL;
const uint32_t MAX_BEAT_MS = 1500UL;
// Clear an old HR after 1.7 s without an accepted beat.
// Slightly longer than the slowest accepted interval (1.5 s).
const uint32_t NO_BEAT_TIMEOUT_MS = 1700UL;

bool amplitudeOK = false;

// Two seconds to settle the filter, then two-second percentile windows.
const uint16_t WARMUP_SAMPLES = 100;
const uint16_t WINDOW_SAMPLES = 100;
// Provisional filtered ADC range: calibrate with real sensor recordings.
// This amplitude check alone cannot identify all movement / bad contact.
const float MIN_FILTERED_RANGE = 100.0f;
const float HYSTERESIS_FRACTION = 0.05f;

// ---------------- BLUETOOTH DATA ----------------
BluetoothSerial SerialBT;
const char* BT_NAME = "AbiESP32";   // host connects to this device name
bool bluetoothReady = false;

struct BtPacket {
  uint32_t sequence;    // sent modulo 65536 (the host wraps at 65536)
  float hr;             // 0 = no valid pulse
  uint8_t recording;
  uint16_t raw[SAMPLES_PER_PACKET];
};

QueueHandle_t btQueue = nullptr;
BtPacket capturePacket = {};
uint16_t packetCount = 0;
uint32_t nextSequence = 0;

// ---------------- ALARM LIMITS (set by the host) ----------------
float lowHrLimit = DEFAULT_LOW_HR_LIMIT;
float highHrLimit = DEFAULT_HIGH_HR_LIMIT;
char limitsLine[24];
uint8_t limitsLineLen = 0;

// ---------------- SCHEDULER AND BUTTON DATA ----------------
uint32_t lastTickTime = 0;
uint32_t lastLcdTime = 0;
bool recording = false;
bool holdUsed = false;
uint32_t pressStart = 0;
bool buzzerEnabled = true;
bool buzzerOn = false;

// ---------------- FILTER AND THRESHOLD DATA ----------------
float hpB, hpA, lpB, lpA;
float previousRaw = 0, previousHp = 0, previousLp = 0;
bool filterStarted = false;
uint16_t warmupCount = 0;
float sampleWindow[WINDOW_SAMPLES];
float sortedWindow[WINDOW_SAMPLES];
uint16_t windowCount = 0;
float adaptiveThreshold = 0;
float upperThreshold = 0, lowerThreshold = 0;
bool signalValid = false;
bool haveThreshold = false;
bool armed = false;
float previousFiltered = 0;

// ---------------- HEART RATE AND LED DATA ----------------
bool havePreviousBeat = false;
uint32_t lastBeatTime = 0;
float heartRate = 0;
// The host packet carries 0.0 when this is false (no valid pulse).
// Raw PPG samples are still sent when HR is invalid.
bool hrValid = false;
bool redOn = false;
uint32_t flashStart = 0;
char previousRows[2][17] = {{0}, {0}};

// Helpers keep each scheduled task easy to follow.
void startupSequence();
void updateButton(uint32_t now);
float bandPass(float raw);
int compareFloats(const void* a, const void* b);
float percentile(float fraction);
void updateThresholds();
void resetBeats();
void detectBeat(float filtered, bool thresholdsChanged, uint32_t now);
void updateAlarm();
void readHostLimits();
void writeRow(uint8_t row, const char* text);
void updateLcd();
void bufferBluetoothSample(uint16_t raw);
void bluetoothSender(void* parameter);

void setup() {
  pinMode(RED_LED_PIN, OUTPUT);
  pinMode(BLUE_LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(SWITCH_PIN, INPUT_PULLUP);
  pinMode(SENSOR_PIN, INPUT);
  digitalWrite(RED_LED_PIN, LOW);
  digitalWrite(BLUE_LED_PIN, LOW);
  digitalWrite(BUZZER_PIN, LOW);
  Serial.begin(115200);
  analogReadResolution(12);
  analogSetPinAttenuation(SENSOR_PIN, ADC_11db);

  // Same first-order high-pass + low-pass as bandpassadapt.
  // HP_HZ and LP_HZ are nominal cutoffs, not sharp frequency boundaries.
  float k = tanf(PI * HP_HZ / SAMPLE_HZ);
  hpB = 1.0f / (1.0f + k);
  hpA = (k - 1.0f) / (k + 1.0f);
  k = tanf(PI * LP_HZ / SAMPLE_HZ);
  lpB = k / (1.0f + k);
  lpA = (k - 1.0f) / (k + 1.0f);

  lcd.begin(16, 2);
  startupSequence();

  // Start Bluetooth once per boot, after the startup delays.
  btQueue = xQueueCreate(1, sizeof(BtPacket));
  if (btQueue != nullptr && SerialBT.begin(BT_NAME)) {
    bluetoothReady = true;
    // Send from a separate task so Bluetooth writes never block sampling.
    if (xTaskCreatePinnedToCore(bluetoothSender, "BTsender", 6144,
                                nullptr, 1, nullptr, 0) != pdPASS) {
      bluetoothReady = false;
      SerialBT.end();
    }
  }

  memset(previousRows, 0, sizeof(previousRows));
  lcd.clear();
  updateLcd();
  // Start scheduling AFTER the startup delays.
  lastLcdTime = millis();
  lastTickTime = micros();
}

void loop() {
  // Lab W3 micros() scheduler: all tasks below run once per 20 ms.
  uint32_t timeUs = micros();
  uint32_t elapsed = timeUs - lastTickTime;
  if (elapsed < TICK_20MSEC) return;

  if (elapsed >= 2 * TICK_20MSEC) {
    // A missed full tick breaks filter timing. Restart rather than
    // making up catch-up samples from repeated ADC reads.
    lastTickTime = timeUs;
    filterStarted = false;
    warmupCount = 0;
    windowCount = 0;
    signalValid = false;
    amplitudeOK = false;
    haveThreshold = false;
    resetBeats();
    // Discard the partial packet; the skipped sequence number lets the
    // host see that time passed.
    packetCount = 0;
    nextSequence++;
    // Unobserved button time must not count towards a continuous hold.
    pressStart = millis();
  }
  else {
    lastTickTime += TICK_20MSEC;
  }
  uint32_t now = millis();

  // 0. Pick up any alarm limits sent by the host.
  readHostLimits();

  // 1. Sample/debounce button and update blue recording LED.
  updateButton(now);
  digitalWrite(BLUE_LED_PIN, recording ? HIGH : LOW);

  // 2. Capture RAW PPG once, then filter a copy for detection.
  uint16_t raw = analogRead(SENSOR_PIN);
  float filtered = bandPass((float)raw);
  // Keep the last second of raw samples for the Bluetooth packet.
  bufferBluetoothSample(raw);

  // 3. Recalculate P10/P90 threshold after each 100 filtered samples.
  bool thresholdsChanged = false;
  if (warmupCount < WARMUP_SAMPLES) {
    warmupCount++;
  }
  else {
    sampleWindow[windowCount++] = filtered;
    if (windowCount >= WINDOW_SAMPLES) {
      updateThresholds();
      windowCount = 0;
      thresholdsChanged = true;
    }
  }

  // Use the percentile amplitude check; clipping does not reset timing.
  // Saturated samples still appear unchanged in RawPPG for inspection.
  signalValid = amplitudeOK;
  if (!signalValid) resetBeats();

  // 4. Find rising crossings and calculate BPM from beat intervals.
  detectBeat(filtered, thresholdsChanged, now);
  if (havePreviousBeat && now - lastBeatTime > NO_BEAT_TIMEOUT_MS) {
    resetBeats();  // Do not keep displaying an old HR after pulse loss.
  }

  // 5. Flash only when BPM is valid; keep LED off while HR shows ---.
  if (!signalValid || !hrValid ||
      (redOn && now - flashStart >= FLASH_MS)) {
    redOn = false;
  }
  digitalWrite(RED_LED_PIN, redOn ? HIGH : LOW);

  // 6. Sound an enabled alarm only for a valid out-of-range HR.
  updateAlarm();

  // 7. Continuous USB Plotter output, independent of recording state.
  // Threshold is in FILTERED ADC units, so include the filtered trace
  // to show the actual detection level. RawPPG stays completely raw.
  if (hrValid) {
    Serial.printf(
      "RawPPG:%u\tFiltered:%.1f\tAdaptiveThreshold:%.1f\tHR:%.1f\t"
      "LowLimit:%.1f\tHighLimit:%.1f\n",
      raw, filtered, adaptiveThreshold, heartRate, lowHrLimit, highHrLimit
    );
  }
  else {
   Serial.printf(
      "RawPPG:%u\tFiltered:%.1f\tAdaptiveThreshold:%.1f\tHR:---\t"
      "LowLimit:%.1f\tHighLimit:%.1f\n",
      raw, filtered, adaptiveThreshold, lowHrLimit, highHrLimit
    );
}

  // 8. Refresh LCD four times per second, without clearing/flickering.
  if (now - lastLcdTime >= LCD_MS) {
    lastLcdTime = now;
    updateLcd();
  }
}

void updateButton(uint32_t now) {
  bool pressed = (digitalRead(SWITCH_PIN) == LOW);
  if (switch1.update(pressed)) {
    if (switch1.state()) {
      pressStart = now;
      holdUsed = false;
    }
    else if (!holdUsed) {
      // Wait for release to distinguish a short press from a hold.
      recording = !recording;
    }
  }

  if (!pressed) pressStart = now;  // Any sampled release breaks the hold.
  if (pressed && switch1.state() && !holdUsed &&
      now - pressStart >= HOLD_MS) {
    holdUsed = true;
    buzzerEnabled = !buzzerEnabled;
    updateLcd();  // Show the sound indicator immediately after the 3 s hold.
  }
}

float bandPass(float raw) {
  if (!filterStarted) {
    // Avoid interpreting the initial sensor DC level as a heartbeat.
    previousRaw = raw;
    previousHp = 0;
    previousLp = 0;
    filterStarted = true;
  }
  float hp = hpB * (raw - previousRaw) - hpA * previousHp;
  float lp = lpB * (hp + previousHp) - lpA * previousLp;
  previousRaw = raw;
  previousHp = hp;
  previousLp = lp;
  return lp;
}

int compareFloats(const void* a, const void* b) {
  float x = *(const float*)a;
  float y = *(const float*)b;
  return (x > y) - (x < y);
}

float percentile(float fraction) {
  // Interpolate between neighbouring values in the sorted window.
  float position = fraction * (WINDOW_SAMPLES - 1);
  uint16_t index = (uint16_t)position;
  if (index >= WINDOW_SAMPLES - 1) return sortedWindow[WINDOW_SAMPLES - 1];
  return sortedWindow[index] + (position - index) *
         (sortedWindow[index + 1] - sortedWindow[index]);
}

void updateThresholds() {
  memcpy(sortedWindow, sampleWindow, sizeof(sampleWindow));
  qsort(sortedWindow, WINDOW_SAMPLES, sizeof(float), compareFloats);
  float p10 = percentile(0.10f);
  float p90 = percentile(0.90f);
  float range = p90 - p10;
  adaptiveThreshold = (p10 + p90) * 0.5f;

  // Small upper/lower separation prevents repeated noise crossings.
  upperThreshold = adaptiveThreshold + HYSTERESIS_FRACTION * range;
  lowerThreshold = adaptiveThreshold - HYSTERESIS_FRACTION * range;
  haveThreshold = true;
  amplitudeOK = (range >= MIN_FILTERED_RANGE);
  if (!amplitudeOK) resetBeats();
}

void resetBeats() {
  havePreviousBeat = false;
  heartRate = 0;  // Internal placeholder only: hrValid=false means MISSING HR.
  hrValid = false;
  armed = false;
  redOn = false;
}

void detectBeat(float filtered, bool thresholdsChanged, uint32_t now) {
  if (signalValid) {
    // Fall below the lower level before accepting an upward crossing.
    if (filtered <= lowerThreshold) armed = true;
    if (!thresholdsChanged && armed &&
        previousFiltered < upperThreshold && filtered >= upperThreshold) {
      armed = false;
      bool accepted = false;
      if (!havePreviousBeat) {
        lastBeatTime = now;
        havePreviousBeat = true;
        accepted = true;  // First beat sets timing; LED waits for valid BPM.
      }
      else {
        uint32_t interval = now - lastBeatTime;
        if (interval >= MIN_BEAT_MS && interval <= MAX_BEAT_MS) {
          heartRate = 60000.0f / (float)interval;
          hrValid = true;
          lastBeatTime = now;
          accepted = true;
        }
        else {
          // Outside 40-150 BPM: hide HR and start a new reference.
          // Updating the reference for short intervals avoids combining
          // successive fast pulses into a falsely plausible slower rate.
          lastBeatTime = now;
          heartRate = 0;
          hrValid = false;
          redOn = false;
        }
      }
      if (accepted) {
        redOn = true;
        flashStart = now;
      }
    }
  }
  // Skip threshold-update crossings so moving a level cannot make a beat.
  previousFiltered = filtered;
}

void readHostLimits() {
  // Bounded work per tick so the 20 ms sampling loop is never held up.
  for (int i = 0; i < 32 && bluetoothReady && SerialBT.available(); i++) {
    char c = (char)SerialBT.read();
    if (c == '$') {
      limitsLineLen = 0;  // a new message always starts here
    }
    if (c == '\n') {
      float low, high;
      if (parseLimits(limitsLine, limitsLineLen, &low, &high)) {
        lowHrLimit = low;
        highHrLimit = high;
      }
      limitsLineLen = 0;
    }
    else if (limitsLineLen < sizeof(limitsLine)) {
      limitsLine[limitsLineLen++] = c;
    }
    else {
      limitsLineLen = 0;  // too long to be ours; wait for the next '$'
    }
  }
}

void updateAlarm() {
  bool valid = signalValid && hrValid;
  bool outsideLimits = heartRate < lowHrLimit || heartRate > highHrLimit;
  // Normal HR and signal loss must not change the user's sound setting.
  bool sound = valid && outsideLimits && buzzerEnabled;
  if (sound && !buzzerOn) {
    tone(BUZZER_PIN, BUZZER_HZ);
    buzzerOn = true;
  }
  else if (!sound && buzzerOn) {
    noTone(BUZZER_PIN);
    digitalWrite(BUZZER_PIN, LOW);
    buzzerOn = false;
  }
}

void writeRow(uint8_t row, const char* text) {
  // Pad to 16 characters so shorter text overwrites the previous row.
  char padded[17];
  memset(padded, ' ', 16);
  padded[16] = '\0';
  size_t length = strlen(text);
  if (length > 16) length = 16;
  memcpy(padded, text, length);
  if (strcmp(padded, previousRows[row]) != 0) {
    lcd.setCursor(0, row);
    lcd.print(padded);
    memcpy(previousRows[row], padded, sizeof(padded));
  }
}

void updateLcd() {
  char row[17];
  if (signalValid && hrValid) {
    const char* status = heartRate < lowHrLimit ? "LOW" :
                         heartRate > highHrLimit ? "HIGH" : "";
    snprintf(row, sizeof(row), "HR:%5.1fBPM %s", heartRate, status);
  }
  else {
    snprintf(row, sizeof(row), "%s", haveThreshold ? "HR: ---" : "HR: WAIT");
  }
  writeRow(0, row);
  // Exactly 16 characters: ! in column 16 means buzzer sound is disabled.
  writeRow(1, !bluetoothReady ? (buzzerEnabled ? "Bluetooth: ERR  " : "Bluetooth: ERR !")
            : SerialBT.hasClient() ? (buzzerEnabled ? "Bluetooth: YES  " : "Bluetooth: YES !")
                                   : (buzzerEnabled ? "Bluetooth: NO   " : "Bluetooth: NO  !"));
}

void bufferBluetoothSample(uint16_t raw) {
  capturePacket.raw[packetCount++] = raw;
  if (packetCount < SAMPLES_PER_PACKET) return;

  capturePacket.sequence = nextSequence++;
  capturePacket.hr = (signalValid && hrValid) ? heartRate : 0.0f;
  capturePacket.recording = recording ? 1 : 0;
  // The queue copies the packet, so the capture buffer is reused at once.
  // Overwrite keeps only the newest packet if the sender falls behind.
  if (bluetoothReady) xQueueOverwrite(btQueue, &capturePacket);
  packetCount = 0;
}

void bluetoothSender(void* parameter) {
  BtPacket packet;
  char frame[PACKET_BUF_SIZE];
  for (;;) {
    if (xQueueReceive(btQueue, &packet, portMAX_DELAY) != pdTRUE) continue;
    if (!SerialBT.hasClient()) continue;

    size_t len = buildPacket(frame, sizeof(frame),
                             (uint16_t)(packet.sequence & 0xFFFF),
                             packet.hr, packet.recording != 0, packet.raw);
    if (len != PACKET_LEN) continue;

    // A slow write waits here, outside the sampling loop.
    size_t sent = SerialBT.write((const uint8_t*)frame, PACKET_LEN);
    if (sent != PACKET_LEN && SerialBT.hasClient()) {
      // End a partial line so the host rejects it and resynchronises.
      SerialBT.write((uint8_t)'\n');
    }
  }
}

void startupSequence() {
  // Show the name immediately on the bottom row, one column left.
  lcd.setCursor(3, 1);
  lcd.print("LumiPulse");

  // Keep both LEDs steadily ON for 3 seconds, starting with the tune.
  // These are the only delay() calls: sampling has not started yet.
  digitalWrite(RED_LED_PIN, HIGH);
  digitalWrite(BLUE_LED_PIN, HIGH);
  const uint16_t notes[5] = {1047, 1319, 1568, 2093, 2637};
  for (uint8_t i = 0; i < 5; i++) {
    tone(BUZZER_PIN, notes[i]);
    delay(100);
    noTone(BUZZER_PIN);
    delay(100);
  }
  delay(2000);  // 1 second of music + 2 seconds = 3 seconds total.
  digitalWrite(RED_LED_PIN, LOW);
  digitalWrite(BLUE_LED_PIN, LOW);
}
