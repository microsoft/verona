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

  class BPState
  {
    uintptr_t bits = (uintptr_t)Priority::Normal;

    BPState(uintptr_t bits_) noexcept : bits(bits_) {}

    friend inline BPState operator|(BPState, Priority);
    friend inline BPState operator|(BPState, Cown*);
    friend inline BPState operator|(BPState, bool);

  public:
    enum struct Mask : uint8_t
    {
      Priority = 0b11,
      PriorityHigh = 0b10,
      HasToken = bits::next_pow2_const((size_t)Priority),
      All = (HasToken << 1) - 1,
    };

    BPState() noexcept {}

    Cown* blocker() const
    {
      return (Cown*)(bits & ~(uintptr_t)Mask::All);
    }

    Priority priority() const
    {
      return (Priority)(bits & (uintptr_t)Mask::Priority);
    }

    bool high_priority() const
    {
      return (bool)(bits & (uintptr_t)Mask::PriorityHigh);
    }

    bool has_token() const
    {
      return (bool)(bits & (uintptr_t)Mask::HasToken);
    }
  };

  inline BPState operator|(BPState state, Priority priority)
  {
    return BPState(state.bits | (uintptr_t)priority);
  }

  inline BPState operator|(BPState state, Cown* blocker)
  {
    assert((state.bits & ~(uintptr_t)BPState::Mask::All) == 0);
    assert(((uintptr_t)blocker & (uintptr_t)BPState::Mask::All) == 0);
    return BPState(state.bits | (uintptr_t)blocker);
  }

  inline BPState operator|(BPState state, bool has_token)
  {
    const size_t shift =
      bits::next_pow2_bits_const((size_t)BPState::Mask::Priority);
    return state.bits | ((uintptr_t)has_token << shift);
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
