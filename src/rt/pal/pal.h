#pragma once

#include "pal_linux.h"

namespace verona::rt
{
#if defined(__linux__)
  using DefaultTCPSocket = LinuxTCPSocket;
#else
  error Unsupported platform
#endif
}
