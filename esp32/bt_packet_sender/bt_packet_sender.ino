/*
** Bluetooth packet sender for the Pulse Sensor System (BMET2922 Major Project).
**
** Samples the PPG every 20 ms (50 Hz) using the course micros() scheduler and,
** every 50 samples (1 s), sends one fixed-length text packet over Bluetooth
** SPP containing: sequence number, MCU pulse rate, recording state and the
** 50 raw samples, plus an XOR checksum (Reqts 1, 15, 18). See packet.h.
**
** Without a pulse sensor, set USE_REAL_SENSOR to 0: the sketch replays the
** course recording (pulse_data.h) so the host can be tested end to end.
**
** Communication status for the MCU (Bianca -> Abi interface): btConnected is
** set/cleared from the Bluetooth SPP open/close events and drives
** BT_LED_PIN (on = host connected). Abi's code can use btConnected for the
** OLED / buzzer as well.
**
** Integration points for Abi's code:
**   readSample()        - replace/extend with the real sensor read
**   getMcuHeartRate()   - return the MCU pulse rate (BPM, 0 if none)
**   recording           - set from the debounced, latching button
**
** Based on the course examples scheduler.ino,
** SerialToSerialBTslaveDemo.ino and
** packet_construction_3.ino.
**
** Author: Bianca Dibden
** Date:   15-Oct-2026
*/

#include "BluetoothSerial.h"
#include "switch.h"
#include "packet.h"
#include "pulse_data.h"

#if !defined(CONFIG_BT_ENABLED) || !defined(CONFIG_BLUEDROID_ENABLED)
#error Bluetooth is not enabled! Use a classic ESP32 board (ESP32 Dev Module).
#endif

// ----------------------------------------------------------------------------
// Configuration
// ----------------------------------------------------------------------------
#define USE_REAL_SENSOR 0          // 0 = replay pulse_data.h, 1 = read the PPG sensor
#define PRINT_PACKETS_TO_USB 1     // echo packets to Serial Monitor (set 0 when using Serial Plotter)

const char *BT_NAME = "BEE-ESP32-XX";   // make unique, e.g. end with the board's serial number

#define SENSOR_PIN 25              // PPG "S" wire (Lab 3)
#define BT_LED_PIN 27              // on while a host is connected
#define BUTTON_PIN 32              // recording on/off (latching)
#define DEBOUNCE_CNT 5             // Reqt 3

#define TICK_20MSEC 20000UL        // 20 ms in microseconds (Reqt 1)
#define MAX_TICKS_BEHIND 5         // if the loop stalls longer than this, resync the clock

// ----------------------------------------------------------------------------
// State
// ----------------------------------------------------------------------------
BluetoothSerial SerialBT;
Switch button(1, DEBOUNCE_CNT);

unsigned long lastTickTime;
uint16_t samples[SAMPLES_PER_PACKET];
uint8_t sampleCount = 0;
uint16_t seq = 0;                  // wraps 65535 -> 0, matching the host
bool recording = false;
char packetBuf[PACKET_BUF_SIZE];

volatile bool btConnected = false; // written by the Bluetooth callback
bool lastReportedConnected = false;

uint16_t replayIndex = 0;
unsigned long packetsSent = 0;
unsigned long packetsNotSent = 0;

// ----------------------------------------------------------------------------
// Bluetooth status
// ----------------------------------------------------------------------------
void btCallback(esp_spp_cb_event_t event, esp_spp_cb_param_t *param)
/*
** Called by the Bluetooth stack on SPP events. Kept minimal (it runs in the
** Bluetooth task): it only records whether a host is connected.
**
** Author: Bianca Dibden
*/
{
  (void)param;   // connection details not needed
  if (event == ESP_SPP_SRV_OPEN_EVT) {
    btConnected = true;
  } else if (event == ESP_SPP_CLOSE_EVT) {
    btConnected = false;
  }
}

void reportConnectionChanges()
/*
** Prints connect/disconnect messages from the main loop (not the callback)
** and updates the connection LED.
**
** Author: Bianca Dibden
*/
{
  bool now = btConnected;
  digitalWrite(BT_LED_PIN, now ? HIGH : LOW);
  if (now != lastReportedConnected) {
    Serial.printf("[BT] Host %s (packets sent %lu, not sent %lu)\n",
                  now ? "connected" : "disconnected", packetsSent, packetsNotSent);
    lastReportedConnected = now;
  }
}

// ----------------------------------------------------------------------------
// Acquisition hooks
// ----------------------------------------------------------------------------
uint16_t readSample()
/*
** Returns the next PPG sample (12-bit ADC count).
** Replay mode steps through pulse_data.h and loops.
**
** Author: Bianca Dibden
*/
{
#if USE_REAL_SENSOR
  return analogRead(SENSOR_PIN);
#else
  uint16_t v = PULSE_DATA[replayIndex];
  replayIndex = (replayIndex + 1) % PULSE_DATA_LEN;
  return v;
#endif
}

float getMcuHeartRate()
/*
** Returns the MCU pulse rate in BPM (0 if no valid pulse).
** Replay mode returns the reference rate stored with the recording.
** TODO (Abi): return the adaptive-threshold MCU pulse rate (Reqt 2).
**
** Author: Bianca Dibden
*/
{
#if USE_REAL_SENSOR
  return 0.0f;
#else
  uint16_t i = (replayIndex + PULSE_DATA_LEN - 1) % PULSE_DATA_LEN;
  return PULSE_RATE_X10[i] / 10.0f;
#endif
}

// ----------------------------------------------------------------------------
// Packet sending
// ----------------------------------------------------------------------------
void sendPacket()
/*
** Builds the packet from the last 50 samples and sends it over Bluetooth.
** If no host is connected the packet is not sent, but the sequence number
** still advances so the host can see how much time passed.
**
** Author: Bianca Dibden
*/
{
  size_t len = buildPacket(packetBuf, sizeof(packetBuf), seq, getMcuHeartRate(),
                           recording, samples);
  if (len != PACKET_LEN) {
    Serial.printf("[PKT] build error: %u bytes\n", (unsigned)len);
  } else if (btConnected) {
    SerialBT.write((const uint8_t *)packetBuf, len);
    packetsSent++;
  } else {
    packetsNotSent++;
  }

#if PRINT_PACKETS_TO_USB
  Serial.write((const uint8_t *)packetBuf, len);
#endif
  seq++;
}

// ----------------------------------------------------------------------------
// Arduino entry points
// ----------------------------------------------------------------------------
void setup()
/*
** Initialise pins, USB serial and Bluetooth SPP.
**
** Author: Bianca Dibden
*/
{
  pinMode(BT_LED_PIN, OUTPUT);
  pinMode(BUTTON_PIN, INPUT);
#if USE_REAL_SENSOR
  pinMode(SENSOR_PIN, INPUT);
#endif

  Serial.begin(115200);
  delay(500);
  Serial.println("\n*** Pulse Sensor System - Bluetooth packet sender ***");
  Serial.printf("Mode: %s\n", USE_REAL_SENSOR ? "real sensor" : "replaying pulse.csv");

  SerialBT.register_callback(btCallback);
  if (!SerialBT.begin(BT_NAME)) {
    Serial.println("[BT] Error initialising Bluetooth");
  } else {
    Serial.printf("[BT] Advertising as \"%s\" - pair from the host\n", BT_NAME);
  }

  lastTickTime = micros();
}

void loop()
/*
** 20 ms scheduler: read button and sensor every tick; send a packet every
** 50 samples (1 s).
**
** Author: Bianca Dibden
*/
{
  unsigned long now = micros();
  if (now - lastTickTime < TICK_20MSEC) {
    return;
  }

  // Advance by exactly one tick (not "= now") so timing errors do not
  // accumulate; if the loop stalled for a long time, resync instead of
  // bursting through many catch-up ticks.
  if (now - lastTickTime > MAX_TICKS_BEHIND * TICK_20MSEC) {
    lastTickTime = now;
  } else {
    lastTickTime += TICK_20MSEC;
  }

  // ---- 20 ms tasks ----
  if (button.update(digitalRead(BUTTON_PIN)) && button.state()) {
    recording = !recording;            // latching: toggle on each press (Lab 3)
    Serial.printf("[BTN] Recording %s\n", recording ? "ON" : "OFF");
  }

  samples[sampleCount++] = readSample();

  // ---- 1 s tasks (every 50 samples) ----
  if (sampleCount >= SAMPLES_PER_PACKET) {
    sendPacket();
    sampleCount = 0;
  }

  reportConnectionChanges();
}
