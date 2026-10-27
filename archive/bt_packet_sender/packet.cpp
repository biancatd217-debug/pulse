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
