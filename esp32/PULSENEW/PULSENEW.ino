/*
** LumiPulse V2 + Bluetooth SPP
**
** Red LED: GPIO17
** Blue LED: GPIO26
** Button: GPIO33 to GND
** PPG signal: GPIO34, power from 3V3
** Buzzer: GPIO27
** LCD: RS=23, E=22, D4=21, D5=19, D6=18, D7=25
**
** Bluetooth name: AbiESP32
** Packet: 50 raw samples plus HR/state/sequence each second, built by
** buildPacket() in packet.cpp so it matches host/packet_format.py exactly
** (276-byte '$...*CC' text packet, see packet.h).
**
** The host GUI sets the HR alarm limits with a '$LIM,LLL,UUU*CC' command.
** They default to 40/200 BPM at power-up, drive the buzzer and LCD, and
** are echoed in every packet so the host can confirm they were received.
**
** Keep switch.h, switch.cpp, packet.h and packet.cpp in this folder.
*/

#include <Arduino.h>
#include "BluetoothSerial.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include <LiquidCrystal.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "switch.h"
#include "packet.h"

#if !defined(CONFIG_BT_SPP_ENABLED)
#error This sketch requires a classic ESP32 with Bluetooth SPP enabled.
#endif

// ======================================================
// PINS
// ======================================================

#define RED_LED_PIN 17
#define BLUE_LED_PIN 26
#define SWITCH_PIN 33
#define SENSOR_PIN 34
#define BUZZER_PIN 27

LiquidCrystal lcd(23, 22, 21, 19, 18, 25);

// ======================================================
// BLUETOOTH
// ======================================================

BluetoothSerial SerialBT;

const char* BT_NAME = "AbiESP32";
bool bluetoothReady = false;

const uint16_t PACKET_SAMPLES = SAMPLES_PER_PACKET;   // 50, from packet.h

struct BtPacket {
  uint32_t sequence;    // sent modulo 65536 (host wraps at 65536)
  float hr;             // 0 = no valid pulse
  uint8_t recording;
  uint16_t raw[PACKET_SAMPLES];
};

QueueHandle_t btQueue = nullptr;

BtPacket capturePacket = {};
uint16_t packetCount = 0;
uint32_t nextSequence = 0;

// ======================================================
// BUTTON AND RECORDING
// ======================================================

Switch switch1(1, 5);

// Sent to host; host must implement actual file saving.
bool recording = false;

// ======================================================
// SAMPLING AND FILTER
// ======================================================

const uint32_t SAMPLE_US = 20000UL;
const float SAMPLE_HZ = 50.0f;

uint32_t lastSampleUs = 0;

const float HP_HZ = 0.3f;
const float LP_HZ = 8.0f;

float hpB, hpA;
float lpB, lpA;

float previousRaw = 0;
float previousHp = 0;
float previousLp = 0;

bool filterStarted = false;

const uint16_t WARMUP_SAMPLES = 100;
uint16_t warmupCount = 0;

// ======================================================
// PERCENTILE THRESHOLD
// ======================================================

const uint16_t WINDOW_SAMPLES = 100;

float sampleWindow[WINDOW_SAMPLES];
float sortedWindow[WINDOW_SAMPLES];

uint16_t windowCount = 0;

// Provisional value: calibrate using recorded signals.
const float MIN_FILTERED_RANGE = 100.0f;
const float HYSTERESIS_FRACTION = 0.05f;

float robustRange = 0;
float threshold = 0;
float upperThreshold = 0;
float lowerThreshold = 0;

// Amplitude check only, not complete signal validation.
bool amplitudeOK = false;
bool haveThresholdWindow = false;

bool armed = false;
float previousDetectionSample = 0;

// ======================================================
// HEART RATE
// ======================================================

const uint32_t MIN_BEAT_MS = 250UL;
const uint32_t MAX_BEAT_MS = 3000UL;

bool havePreviousBeat = false;
uint32_t lastBeatMs = 0;

float heartRate = 0;
bool hrValid = false;

// ======================================================
// LED, BUZZER AND LCD TIMING
// ======================================================

const uint32_t FLASH_MS = 100UL;
bool redOn = false;
uint32_t flashStartMs = 0;

// HR alarm limits (BPM). Power-up defaults; the host GUI changes them with a
// $LIM command, handled in the Bluetooth task. Single 16-bit writes, so the
// sampling loop always reads a whole value.
volatile uint16_t lowHr = 40;
volatile uint16_t highHr = 200;
const uint16_t BUZZER_HZ = 2000;

bool buzzerOn = false;

const uint32_t LCD_MS = 250UL;
uint32_t lastLcdMs = 0;

char previousRows[2][17] = {{0}, {0}};

// ======================================================
// FUNCTION DECLARATIONS
// ======================================================

void startupSequence();
void resetBeats();
float bandPass(float raw);
int compareFloats(const void* a, const void* b);
float percentile(float fraction);
void updateThresholds();
void writeRow(uint8_t row, const char* text);
void updateLcd();

void bufferBluetoothSample(uint16_t raw);
void bluetoothSender(void* parameter);
void serviceBluetoothCommands();

// ======================================================
// SETUP
// ======================================================

void setup() {
  pinMode(RED_LED_PIN, OUTPUT);
  pinMode(BLUE_LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(SENSOR_PIN, INPUT);
  pinMode(SWITCH_PIN, INPUT_PULLUP);

  digitalWrite(RED_LED_PIN, LOW);
  digitalWrite(BLUE_LED_PIN, LOW);
  digitalWrite(BUZZER_PIN, LOW);

  Serial.begin(115200);

  analogReadResolution(12);
  analogSetPinAttenuation(SENSOR_PIN, ADC_11db);

  float k = tanf(PI * HP_HZ / SAMPLE_HZ);
  hpB = 1.0f / (1.0f + k);
  hpA = (k - 1.0f) / (k + 1.0f);

  k = tanf(PI * LP_HZ / SAMPLE_HZ);
  lpB = k / (1.0f + k);
  lpA = (k - 1.0f) / (k + 1.0f);

  lcd.begin(16, 2);
  startupSequence();

  // Initialise Bluetooth once per boot.
  btQueue = xQueueCreate(1, sizeof(BtPacket));

  if (btQueue != nullptr && SerialBT.begin(BT_NAME)) {
    bluetoothReady = true;

    // Send from a separate task so Bluetooth writes
    // do not block the sampling loop.
    if (xTaskCreatePinnedToCore(
          bluetoothSender,
          "BTsender",
          6144,
          nullptr,
          1,
          nullptr,
          0
        ) != pdPASS) {

      bluetoothReady = false;
      SerialBT.end();
    }
  }

  lcd.clear();
  memset(previousRows, 0, sizeof(previousRows));
  updateLcd();

  lastLcdMs = millis();
  lastSampleUs = micros();
}

// ======================================================
// STARTUP
// ======================================================

void startupSequence() {
  lcd.clear();

  lcd.setCursor(4, 0);
  lcd.print("LumiPulse");

  lcd.setCursor(2, 1);
  lcd.print("Starting...");

  const uint16_t startupNotes[5] = {
    1047, 1319, 1568, 2093, 2637
  };

  for (uint8_t i = 0; i < 5; i++) {
    tone(BUZZER_PIN, startupNotes[i]);
    delay(100);

    noTone(BUZZER_PIN);
    digitalWrite(BUZZER_PIN, LOW);
    delay(100);
  }

  digitalWrite(RED_LED_PIN, HIGH);
  digitalWrite(BLUE_LED_PIN, HIGH);

  delay(1000);

  digitalWrite(RED_LED_PIN, LOW);
  digitalWrite(BLUE_LED_PIN, LOW);
}

// ======================================================
// MAIN LOOP
// ======================================================

void loop() {
  uint32_t timeUs = micros();
  uint32_t elapsed = timeUs - lastSampleUs;

  if (elapsed < SAMPLE_US) {
    return;
  }

  if (elapsed >= 2 * SAMPLE_US) {
    // Recover after a missed sampling tick.
    lastSampleUs = timeUs;

    filterStarted = false;
    warmupCount = 0;
    windowCount = 0;
    robustRange = 0;

    amplitudeOK = false;
    haveThresholdWindow = false;

    resetBeats();

    // Discard partial packet after a sampling gap.
    packetCount = 0;
    nextSequence++;
  }
  else {
    lastSampleUs += SAMPLE_US;
  }

  uint32_t now = millis();

  // Button and recording LED.
  bool pressed = (digitalRead(SWITCH_PIN) == LOW);

  if (switch1.update(pressed) && switch1.state()) {
    recording = !recording;
  }

  digitalWrite(BLUE_LED_PIN, recording ? HIGH : LOW);

  // Read and filter PPG.
  uint16_t raw = analogRead(SENSOR_PIN);
  float filtered = bandPass((float)raw);

  bool thresholdsChanged = false;

  if (warmupCount < WARMUP_SAMPLES) {
    warmupCount++;
  }
  else {
    sampleWindow[windowCount++] = filtered;

    if (windowCount == WINDOW_SAMPLES) {
      updateThresholds();

      windowCount = 0;
      thresholdsChanged = true;
    }
  }

  // Hysteresis and beat detection.
  if (amplitudeOK) {
    if (filtered <= lowerThreshold) {
      armed = true;
    }

    if (!thresholdsChanged &&
        armed &&
        previousDetectionSample < upperThreshold &&
        filtered >= upperThreshold) {

      armed = false;

      if (!havePreviousBeat) {
        lastBeatMs = now;
        havePreviousBeat = true;
      }
      else {
        uint32_t interval = now - lastBeatMs;

        if (interval >= MIN_BEAT_MS &&
            interval <= MAX_BEAT_MS) {

          heartRate = 60000.0f / (float)interval;
          hrValid = true;
          lastBeatMs = now;

          redOn = true;
          flashStartMs = now;
        }
        else if (interval > MAX_BEAT_MS) {
          lastBeatMs = now;
          heartRate = 0;
          hrValid = false;
          redOn = false;
        }
      }
    }
  }

  previousDetectionSample = filtered;

  // Expire stale HR.
  if (havePreviousBeat &&
      now - lastBeatMs > MAX_BEAT_MS) {
    resetBeats();
  }

  // Red heartbeat LED.
  if (!amplitudeOK ||
      (redOn && now - flashStartMs >= FLASH_MS)) {
    redOn = false;
  }

  digitalWrite(RED_LED_PIN, redOn ? HIGH : LOW);

  // Local HR alarm.
  bool alarm =
      amplitudeOK &&
      hrValid &&
      (heartRate < lowHr || heartRate > highHr);

  if (alarm && !buzzerOn) {
    tone(BUZZER_PIN, BUZZER_HZ);
    buzzerOn = true;
  }
  else if (!alarm && buzzerOn) {
    noTone(BUZZER_PIN);
    digitalWrite(BUZZER_PIN, LOW);
    buzzerOn = false;
  }

  // Collect raw samples for Bluetooth.
  bufferBluetoothSample(raw);

  // USB Serial Plotter remains available.
  Serial.printf(
    "RawPPG:%u\tFiltered:%.1f\tThreshold:%.1f\tUpper:%.1f\tLower:%.1f\tRange:%.1f\tHR:%.1f\n",
    raw,
    filtered,
    threshold,
    upperThreshold,
    lowerThreshold,
    robustRange,
    heartRate
  );

  if (now - lastLcdMs >= LCD_MS) {
    lastLcdMs = now;
    updateLcd();
  }
}

// ======================================================
// DETECTOR HELPERS
// ======================================================

void resetBeats() {
  havePreviousBeat = false;
  hrValid = false;
  heartRate = 0;
  armed = false;
  redOn = false;
}

float bandPass(float raw) {
  if (!filterStarted) {
    previousRaw = raw;
    previousHp = 0;
    previousLp = 0;
    filterStarted = true;
  }

  float hp =
      hpB * (raw - previousRaw) - hpA * previousHp;

  float lp =
      lpB * (hp + previousHp) - lpA * previousLp;

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
  float position = fraction * (WINDOW_SAMPLES - 1);

  uint16_t index = (uint16_t)position;
  float remainder = position - index;

  if (index >= WINDOW_SAMPLES - 1) {
    return sortedWindow[WINDOW_SAMPLES - 1];
  }

  return sortedWindow[index] +
         remainder *
         (sortedWindow[index + 1] - sortedWindow[index]);
}

void updateThresholds() {
  haveThresholdWindow = true;

  memcpy(
    sortedWindow,
    sampleWindow,
    sizeof(sampleWindow)
  );

  qsort(
    sortedWindow,
    WINDOW_SAMPLES,
    sizeof(float),
    compareFloats
  );

  float p10 = percentile(0.10f);
  float p90 = percentile(0.90f);

  robustRange = p90 - p10;
  threshold = (p10 + p90) * 0.5f;

  upperThreshold =
      threshold + HYSTERESIS_FRACTION * robustRange;

  lowerThreshold =
      threshold - HYSTERESIS_FRACTION * robustRange;

  amplitudeOK = (robustRange >= MIN_FILTERED_RANGE);

  if (!amplitudeOK) {
    resetBeats();
  }
}

// ======================================================
// LCD
// ======================================================

void writeRow(uint8_t row, const char* text) {
  char padded[17];

  memset(padded, ' ', 16);
  padded[16] = '\0';

  size_t length = strlen(text);

  if (length > 16) {
    length = 16;
  }

  memcpy(padded, text, length);

  if (strcmp(padded, previousRows[row]) != 0) {
    lcd.setCursor(0, row);
    lcd.print(padded);

    memcpy(previousRows[row], padded, 17);
  }
}

void updateLcd() {
  char row[17];
  const char* status;

  if (!haveThresholdWindow) {
    status = "WAIT";
  }
  else if (!amplitudeOK || !hrValid) {
    status = "NO PULSE";
  }
  else if (heartRate < lowHr) {
    status = "LOW";
  }
  else if (heartRate > highHr) {
    status = "HIGH";
  }
  else {
    status = "OK";
  }

  if (amplitudeOK && hrValid) {
    snprintf(
      row,
      sizeof(row),
      "HR:%5.1f BPM %s",
      heartRate,
      !bluetoothReady
        ? "ER"
        : (SerialBT.hasClient() ? "BT" : "--")
    );
  }
  else {
    snprintf(
      row,
      sizeof(row),
      "HR: -- BPM   %s",
      !bluetoothReady
        ? "ER"
        : (SerialBT.hasClient() ? "BT" : "--")
    );
  }

  writeRow(0, row);

  snprintf(
    row,
    sizeof(row),
    "REC:%s %s",
    recording ? "ON" : "OFF",
    status
  );

  writeRow(1, row);
}

// ======================================================
// BLUETOOTH SAMPLE BUFFER
// ======================================================

void bufferBluetoothSample(uint16_t raw) {
  capturePacket.raw[packetCount++] = raw;

  if (packetCount < PACKET_SAMPLES) {
    return;
  }

  capturePacket.sequence = nextSequence++;
  capturePacket.hr =
      (amplitudeOK && hrValid) ? heartRate : 0.0f;
  capturePacket.recording = recording ? 1 : 0;

  // Queue copies the packet, so the capture buffer
  // can immediately be reused for the next second.
  if (bluetoothReady) {
    xQueueOverwrite(btQueue, &capturePacket);
  }

  packetCount = 0;
}

// ======================================================
// BACKGROUND BLUETOOTH SENDER
// ======================================================

void serviceBluetoothCommands() {
  // Reads any bytes the host has sent and applies a valid
  // "$LIM,LLL,UUU*CC" command to the HR alarm limits. Never blocks.
  // Corrupted, over-long or out-of-range commands are ignored; the host
  // sees that the echoed limits did not change and re-sends.
  static char cmdBuf[24];
  static uint8_t cmdLen = 0;
  static bool cmdOverflow = false;

  int budget = 64;   // bytes per call keeps each pass short

  while (budget-- > 0 && SerialBT.available()) {
    int c = SerialBT.read();

    if (c < 0) {
      break;
    }

    if (c == '\n') {
      uint16_t low, high;

      if (!cmdOverflow &&
          parseLimitsCommand(cmdBuf, cmdLen, &low, &high)) {
        lowHr = low;
        highHr = high;

        Serial.printf("[BT] HR limits set to %u-%u BPM\n",
                      (unsigned)low, (unsigned)high);
      }

      cmdLen = 0;
      cmdOverflow = false;
    }
    else if (cmdLen < sizeof(cmdBuf)) {
      cmdBuf[cmdLen++] = (char)c;
    }
    else {
      cmdOverflow = true;   // too long to be a command; wait for newline
    }
  }
}

void bluetoothSender(void* parameter) {
  // This task owns all Bluetooth reads and writes.
  BtPacket packet;
  char frame[PACKET_BUF_SIZE];

  for (;;) {
    serviceBluetoothCommands();

    // Wake at least every 20 ms to poll for host commands.
    if (xQueueReceive(
          btQueue,
          &packet,
          pdMS_TO_TICKS(20)
        ) != pdTRUE) {
      continue;
    }

    if (!SerialBT.hasClient()) {
      continue;
    }

    size_t len = buildPacket(
      frame,
      sizeof(frame),
      (uint16_t)(packet.sequence & 0xFFFF),
      packet.hr,
      packet.recording != 0,
      lowHr,
      highHr,
      packet.raw
    );

    if (len != PACKET_LEN) {
      continue;
    }

    // A slow Bluetooth write waits here,
    // outside the sensor-sampling loop.
    size_t sent = SerialBT.write(
      (const uint8_t*)frame,
      PACKET_LEN
    );

    if (sent != PACKET_LEN) {
      // Terminate a partial line so the host can
      // reject it and find the next complete packet.
      if (SerialBT.hasClient()) {
        SerialBT.write((uint8_t)'\n');
      }
    }
  }
}
