/*
** Packet construction for the Pulse Sensor System Bluetooth link.
**
** Format (ASCII, fixed length PACKET_LEN = 276 bytes, NMEA-style):
**
**   $SSSSS,HHH.H,R,LLL,UUU,s01,s02,...,s50*CC\n
**
**   SSSSS  sequence number 00000-65535 (wraps)
**   HHH.H  MCU pulse rate, BPM, 1 d.p. (000.0 = no valid pulse)
**   R      recording state 0/1
**   LLL    low HR alarm limit in use, whole BPM
**   UUU    high HR alarm limit in use, whole BPM
**   sNN    50 raw ADC samples, 0000-4095, oldest first
**   CC     XOR of every byte between '$' and '*', 2 upper-case hex digits
**
** Host -> ESP32 command (sets the HR alarm limits):
**
**   $LIM,LLL,UUU*CC\n      (16 bytes; CC = XOR of "LIM,LLL,UUU")
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
#define PACKET_LEN 276
#define LIMITS_CMD_LEN 16                 // including the newline
#define LIMIT_MIN 20                      // BPM
#define LIMIT_MAX 250
#define PACKET_BUF_SIZE (PACKET_LEN + 4)   // room for snprintf's null terminator

size_t buildPacket(char *out, size_t outSize, uint16_t seq, float heartRate,
                   bool recording, uint16_t lowLimit, uint16_t highLimit,
                   const uint16_t *samples);

// Validates a received line "$LIM,LLL,UUU*CC" (with or without the trailing
// "\n"/"\r"). On success stores the limits and returns true; returns false
// for any corrupted, malformed or out-of-range command.
bool parseLimitsCommand(const char *line, size_t len, uint16_t *low, uint16_t *high);

uint8_t packetChecksum(const char *payload, size_t len);

#endif
