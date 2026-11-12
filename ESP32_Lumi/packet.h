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
** Host -> ESP32 alarm limits message (ASCII, 20 bytes): $LIM,LLL.L,HHH.H*CC\n
**
**   LLL.L  low HR limit in BPM, HHH.H high HR limit in BPM (1 d.p.)
**   CC     XOR of the bytes between '$' and '*', as above
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

// Validates one received line (without the newline); true and *low / *high
// set if it is a well-formed limits message with a correct checksum and
// 0 <= low < high <= 999.9.
bool parseLimits(const char *line, size_t len, float *low, float *high);

uint8_t packetChecksum(const char *payload, size_t len);

#endif
