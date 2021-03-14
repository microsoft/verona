// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "pal_linux.h"

namespace verona::rt::io
{
#if defined(__linux__)
  using Event = LinuxEvent;
  using Poller = LinuxPoller;
  using TCP = LinuxTCP;
  template<typename T>
  using Result = LinuxResult<T>;
#else
  error Unsupported platform
#endif
}
