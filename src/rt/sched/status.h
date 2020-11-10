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

  /// Tracks status information for a cown. This class may only be modified by
  /// the scheduler thread running a cown.
  /// TODO: possibly I/O thread when cown is uncheduled
  class alignas(4) Status
  {
    /// Other miscellaneous bits. 6 bits are reserved for future use.
    /// Bit 0 indicates if the token message is in a cown's queue. If zero, a
    /// new token message should be added if the cown runs another message.
    /// Bit 1 indicates if a cown is held in an overloaded state regardless of
    /// its `total_load()`.
    uint8_t _misc = 0;

    template<uint8_t i>
    inline void set_misc(bool value)
    {
      static_assert(i < 8);
      const auto mask = ~((uint8_t)1 << i);
      _misc = (_misc & mask) | ((uint8_t)value << i);
    }

  public:
    /// Return true if this cown is overloaded.
    inline bool overloaded() const
    {
      return bits::extract<1, 1>(_misc);
    }

    /// Set the bit value that holds the overloaded state regardless of the
    /// `total_load()` on a cown.
    inline void set_overloaded(bool value)
    {
      set_misc<1>(value);
    }

    /// Indicates if the token message is in a cown's queue. If zero, a
    /// new token message should be added if the cown runs another message.
    inline bool has_token()
    {
      return bits::extract<0, 0>(_misc) == 1;
    }

    /// Set whether the cown has a token message in its queue.
    inline void set_has_token(bool value)
    {
      set_misc<0>(value);
    }
  };
}
