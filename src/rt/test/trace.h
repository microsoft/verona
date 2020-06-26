// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "aal/aal.h"
#include "log.h"

namespace logger
{
  inline void trace(
    std::string_view prefix,
    uintptr_t label,
    size_t value,
    uint64_t timestamp = snmalloc::Aal::tick())
  {
    cout() << prefix << "," << label << "," << timestamp << "," << value
           << "\n";
  }

  template<typename T>
  inline void trace(
    std::string_view prefix,
    const T* label,
    size_t value,
    uint64_t timestamp = snmalloc::Aal::tick())
  {
    trace(prefix, reinterpret_cast<uintptr_t>(label), value, timestamp);
  }
}
