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

  class BackpressureState
  {
    static constexpr uintptr_t priority_mask = (uintptr_t)0b011;
    static constexpr uintptr_t token_mask = (uintptr_t)0b100;
    static constexpr uintptr_t blocker_mask = ~(token_mask | priority_mask);

    static_assert(priority_mask == (uintptr_t)PriorityMask::All);

    uintptr_t bits = 0;

    BackpressureState(uintptr_t bits_) : bits(bits_) {}

  public:
    BackpressureState() {}

    Cown* blocker() const
    {
      return (Cown*)(bits & blocker_mask);
    }

    Priority priority() const
    {
      return (Priority)(bits & priority_mask);
    }

    bool priority(PriorityMask mask) const
    {
      return priority() & mask;
    }

    bool token() const
    {
      return (bits & token_mask) != 0;
    }

    BackpressureState with_blocker(const Cown* blocker) const
    {
      return BackpressureState(
        ((uintptr_t)blocker & blocker_mask) | (bits & ~blocker_mask));
    }

    BackpressureState with_priority(Priority priority) const
    {
      return BackpressureState((uintptr_t)priority | (bits & ~priority_mask));
    }

    BackpressureState with_token(bool token) const
    {
      return BackpressureState(((uintptr_t)token << 2) | (bits & ~token_mask));
    }
  };
}
