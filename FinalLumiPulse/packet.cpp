/*
** Packet construction for the Pulse Sensor System Bluetooth link.
** See packet.h for the format.
**
** Author: Bianca Dibden
** Date:   15-Oct-2026
*/

#include "packet.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

uint8_t packetChecksum(const char *payload, size_t len)
/*
** XOR of len bytes starting at payload (the bytes between '$' and '*').
**
** Author: Bianca Dibden
*/
{
  uint8_t cs = 0;
  for (size_t i = 0; i < len; i++) {
    cs ^= (uint8_t)payload[i];
  }
  return cs;
}

size_t buildPacket(char *out, size_t outSize, uint16_t seq, float heartRate,
                   bool recording, const uint16_t *samples)
/*
** Writes one complete, null-terminated packet into out.
**
** out        - buffer of at least PACKET_BUF_SIZE chars
** seq        - packet sequence number
** heartRate  - MCU pulse rate in BPM (clamped to 0-999.9; 0 = no pulse)
** recording  - recording state from the button
** samples    - SAMPLES_PER_PACKET raw ADC values (clamped to 0-4095)
**
** returns - number of bytes written (PACKET_LEN), or 0 if out is too small.
**
** Every field is zero-padded to a fixed width so all packets are PACKET_LEN
** bytes long (Reqt 18).
**
** Author: Bianca Dibden
*/
{
  if (outSize < PACKET_BUF_SIZE) {
    return 0;
  }
  // Clamp so the fixed-width fields can never overflow (NaN fails both tests -> 0).
  if (!(heartRate >= 0.0f)) heartRate = 0.0f;
  if (heartRate > 999.9f) heartRate = 999.9f;

  int pos = snprintf(out, outSize, "$%05u,%05.1f,%u",
                     (unsigned)seq, heartRate, recording ? 1u : 0u);

  for (int i = 0; i < SAMPLES_PER_PACKET; i++) {
    unsigned s = samples[i] > 4095 ? 4095 : samples[i];
    pos += snprintf(out + pos, outSize - pos, ",%04u", s);
  }

  uint8_t cs = packetChecksum(out + 1, pos - 1);   // skip the '$'
  pos += snprintf(out + pos, outSize - pos, "*%02X\n", cs);
  return (size_t)pos;
}

bool parseAlarm(const char *line, size_t len, uint8_t *state)
/*
** Checks a line of the form "$ALM,s*CC" (newline already removed).
**
** line  - received characters
** len   - number of characters
** state - set to 0-3 on success
**
** returns - true if the format and checksum are correct.
**
** Author: Bianca Dibden
*/
{
  if (len != 9 || line[0] != '$' || line[6] != '*') return false;
  if (memcmp(line + 1, "ALM,", 4) != 0) return false;
  if (line[5] < '0' || line[5] > '3') return false;

  char hex[3] = {line[7], line[8], '\0'};
  char *end = nullptr;
  long cs = strtol(hex, &end, 16);
  if (*end != '\0' || end == hex) return false;
  if ((uint8_t)cs != packetChecksum(line + 1, 5)) return false;

  *state = (uint8_t)(line[5] - '0');
  return true;
}
