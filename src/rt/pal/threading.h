// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#if !defined(VERONA_EXTERNAL_THREADING)

#  include "cpu.h"

#  include <thread>

namespace verona::rt
{
  using PlatformThread = std::thread;
}

#else

#  include <verona_external_threading.h>

#endif