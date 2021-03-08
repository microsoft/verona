#pragma once

#include "pal_linux.h"

namespace verona::rt::io
{
#if defined(__linux__)
  using Event = LinuxEvent;
  using Poller = LinuxPoller;
  using TCP = LinuxTCP;
#else
  error Unsupported platform
#endif
}
