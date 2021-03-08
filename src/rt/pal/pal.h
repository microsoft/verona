#pragma once

#include "pal_linux.h"

namespace verona::rt::io
{
#if defined(__linux__)
  using TCP = LinuxTCP;
  template<typename T>
  using Event = LinuxEvent<T>;
  template<typename T>
  using Poller = LinuxPoller<T>;
#else
  error Unsupported platform
#endif
}
