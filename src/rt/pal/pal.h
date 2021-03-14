// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "pal_linux.h"

namespace verona::rt::io
{
#if defined(__linux__)
#  define PLATFORM_SUPPORTS_IO
  using Event = LinuxEvent;
  using Poller = LinuxPoller;
  using TCP = LinuxTCP;
  template<typename T>
  using Result = LinuxResult<T>;
#else
  // unsuported platforms
#endif
}
