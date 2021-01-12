// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "../ds/morebits.h"

#include <ostream>

namespace verona::rt
{
  /// Backpressure priority level
  enum struct Priority : uint8_t
  {
    /// Cown is muted. A muted cown must not be scheduled.
    Low = 0b01,
    /// Cown is sleeping. A sleeping cown is in a normal state and cannot change
    /// priority until scheduled.
    // Sleeping = 0b001,
    /// Cown is normal and scheduled.
    Normal = 0b00,
    /// Cown is temporarily protected from muting. This state may be reached by
    /// becoming overloaded or by being required for a behaviour with another
    /// high priority cown.
    High = 0b10,
    /// Cown is high priority, but may transition back to normal if another
    /// token message falls out of the queue.
    MaybeHigh = 0b11,
  };

  enum struct PriorityMask : uint8_t
  {
    All = 0b11,
    High = 0b10,
  };

  inline uintptr_t operator|(Cown* blocker, Priority p)
  {
    return (uintptr_t)blocker | (uintptr_t)p;
  }

  constexpr inline bool operator&(Priority p, PriorityMask m)
  {
    return (uint8_t)p & (uint8_t)m;
  }

  inline std::ostream& operator<<(std::ostream& os, Priority p)
  {
    switch (p)
    {
      case Priority::Low:
        return os << "Low";
      case Priority::Normal:
        return os << "Normal";
      case Priority::High:
        return os << "High";
      case Priority::MaybeHigh:
        return os << "MaybeHigh";
      default:
        abort();
    }
  }
}
