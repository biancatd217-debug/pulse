/*
** Packet construction for the Pulse Sensor System Bluetooth link.
**
** Format (ASCII, fixed length PACKET_LEN = 268 bytes, NMEA-style):
**
**   $SSSSS,HHH.H,R,s01,s02,...,s50*CC\n
**
**   SSSSS  sequence number 00000-65535 (wraps)
**   HHH.H  MCU pulse rate, BPM, 1 d.p. (000.0 = no valid pulse)
**   R      recording state 0/1
**   sNN    50 raw ADC samples, 0000-4095, oldest first
**   CC     XOR of every byte between '$' and '*', 2 upper-case hex digits
**
** Host -> ESP32 alarm message (ASCII, 10 bytes): $ALM,s*CC\n
**
**   s   0 normal, 1 low, 2 high, 3 none (no valid HR / no data)
**   CC  XOR of the bytes between '$' and '*', as above
**
** Must match Code/host/packet_format.py - change both together.
**
** Author: Bianca Dibden
** Date:   15-Oct-2026
*/

#ifndef PACKET_H
#define PACKET_H

#include <stdint.h>
#include <stddef.h>

#define SAMPLES_PER_PACKET 50
#define PACKET_LEN 268
#define PACKET_BUF_SIZE (PACKET_LEN + 4)   // room for snprintf's null terminator

size_t buildPacket(char *out, size_t outSize, uint16_t seq, float heartRate,
                   bool recording, const uint16_t *samples);

#define ALARM_NORMAL 0
#define ALARM_LOW 1
#define ALARM_HIGH 2
#define ALARM_NONE 3

// Validates one received line (without the newline); true and *state set if
// it is a well-formed alarm message with a correct checksum.
bool parseAlarm(const char *line, size_t len, uint8_t *state);

uint8_t packetChecksum(const char *payload, size_t len);

#endif
