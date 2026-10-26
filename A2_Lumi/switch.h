/*
** Class to debounce switches.
**
** Author:  Greg Watkins
** Date:    27 Aug 2021
**
*/

#include <stdint.h>

#ifndef  __SWITCH__
class Switch
{
  private:
    uint8_t _id;        // the number of the switch
    uint8_t _threshold; // Threshold for consecutive state mismatch before decidig teh state f teh switch has changed.
    bool    _state;     // Current switch state
    bool    _changed;   // indicates a change of state.
    uint8_t _count;     // count of consecutive state mismatches.
      
  public:
    Switch(uint8_t  id, uint8_t threshold);
    bool update(bool value);
    bool state(void);
    uint8_t id(void);
    bool changed(void);
};
#define __SWITCH__
#endif
