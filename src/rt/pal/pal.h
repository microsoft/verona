#pragma once

#include "pal_linux.h"

namespace verona::rt::io
{
#if defined(__linux__)
  using DefaultTCPSocket = LinuxTCPSocket;
  template<typename T>
  using DefaultPoller = LinuxPoller<T>;
#else
  error Unsupported platform
#endif
}
