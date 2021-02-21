#pragma once

#include "pal_linux.h"

namespace verona::rt::io
{
#if defined(__linux__)
  using DefaultTCPSocket = LinuxTCPSocket;
  using DefaultPoller = LinuxPoller;
#else
  error Unsupported platform
#endif
}
