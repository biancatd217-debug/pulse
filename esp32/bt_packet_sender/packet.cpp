/*
** Packet construction for the Pulse Sensor System Bluetooth link.
** See packet.h for the format.
**
** Author: Bianca Dibden
** Date:   15-Oct-2026
*/

#include "packet.h"
#include <stdio.h>

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
                   bool recording, uint16_t lowLimit, uint16_t highLimit,
                   const uint16_t *samples)
/*
** Writes one complete, null-terminated packet into out.
**
** out        - buffer of at least PACKET_BUF_SIZE chars
** seq        - packet sequence number
** heartRate  - MCU pulse rate in BPM (clamped to 0-999.9; 0 = no pulse)
** recording  - recording state from the button
** lowLimit, highLimit - HR alarm limits in use (BPM, clamped to 0-999)
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

  if (lowLimit > 999) lowLimit = 999;
  if (highLimit > 999) highLimit = 999;

  int pos = snprintf(out, outSize, "$%05u,%05.1f,%u,%03u,%03u",
                     (unsigned)seq, heartRate, recording ? 1u : 0u,
                     (unsigned)lowLimit, (unsigned)highLimit);

  for (int i = 0; i < SAMPLES_PER_PACKET; i++) {
    unsigned s = samples[i] > 4095 ? 4095 : samples[i];
    pos += snprintf(out + pos, outSize - pos, ",%04u", s);
  }

  uint8_t cs = packetChecksum(out + 1, pos - 1);   // skip the '$'
  pos += snprintf(out + pos, outSize - pos, "*%02X\n", cs);
  return (size_t)pos;
}

static bool parseDigits3(const char *p, uint16_t *value)
/*
** Parses exactly three decimal digits.
**
** Author: Bianca Dibden
*/
{
  if (p[0] < '0' || p[0] > '9' || p[1] < '0' || p[1] > '9' || p[2] < '0' || p[2] > '9') {
    return false;
  }
  *value = (uint16_t)((p[0] - '0') * 100 + (p[1] - '0') * 10 + (p[2] - '0'));
  return true;
}

static int hexValue(char c)
/*
** Value of an upper- or lower-case hex digit, or -1.
**
** Author: Bianca Dibden
*/
{
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  return -1;
}

bool parseLimitsCommand(const char *line, size_t len, uint16_t *low, uint16_t *high)
/*
** Validates "$LIM,LLL,UUU*CC": length, framing, command name, checksum, digits
** and range (LIMIT_MIN <= low < high <= LIMIT_MAX). A trailing "\r\n" is
** tolerated. Nothing is written to low/high unless every check passes.
**
** Author: Bianca Dibden
*/
{
  while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
    len--;
  }
  if (len != LIMITS_CMD_LEN - 1 || line[0] != '$' || line[len - 3] != '*') {
    return false;
  }
  const char *payload = line + 1;                     // "LIM,LLL,UUU"
  if (payload[0] != 'L' || payload[1] != 'I' || payload[2] != 'M' ||
      payload[3] != ',' || payload[7] != ',') {
    return false;
  }
  int hi = hexValue(line[len - 2]);
  int lo = hexValue(line[len - 1]);
  if (hi < 0 || lo < 0 || packetChecksum(payload, len - 4) != (uint8_t)(hi * 16 + lo)) {
    return false;
  }
  uint16_t l, u;
  if (!parseDigits3(payload + 4, &l) || !parseDigits3(payload + 8, &u)) {
    return false;
  }
  if (l < LIMIT_MIN || u > LIMIT_MAX || l >= u) {
    return false;
  }
  *low = l;
  *high = u;
  return true;
}
