/*
** Class to debounce switches.
**
** Author:  Greg Watkins
** Date:    27 Aug 2021
**
*/

#include "switch.h"

Switch::Switch(uint8_t id, uint8_t threshold = 10)
/*
** Constructor
** Initialise id and threshold. The default threshold value is 10.
**
*/
{
  _id        = id;
  _threshold = threshold;
  _state     = false;
  _changed   = false;
  _count     = 0;
}

uint8_t Switch::id(void)
/*
** Returns the switch id
*/
{
  return _id;
}

bool Switch::changed(void)
{
/*
** Returns whtehr the switch has changed state with the last update.
*/
  return _changed;
}

bool Switch::state(void)
/*
** returns the current switch state.
*/
{
  return _state;
}

bool Switch::update(bool value)
/*
** Updates the count of switch state mismatches. This method
** shoudl be called each time teh applciation samples the switch input.
**
** value -  the state read from the physical switch.
**
** returns - true if the switch has changed state. 
**          i.e. the threshold of state mismatches has 
**          been exceeded. Otherwise fale
*/
{
  if (value == _state)
  { // have a match. Decrement counter but only to 0
    if (_count > 0)
    {
      _count--;
    }
  }
  else
  {
    // have a mismatch. increment count.
    _count++;
  }

  if (_count >= _threshold)
  {
    _changed = true;
    _state = !_state;
    _count = 0;
  }
  else
    _changed = false;

  return _changed;
}

