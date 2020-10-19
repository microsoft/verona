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

  /// Tracks status information for a cown. This class may only be modified by
  /// the scheduler thread running a cown.
  /// TODO: possibly I/O thread when cown is uncheduled
  class alignas(4) Status
  {
    // Load at which cown is overloaded.
    static constexpr uint32_t overload_threshold = 800;

    /// Ring buffer with a capacity for 4 4-bit entries.
    uint16_t _load_hist = 0;
    /// Tracks an approximation of a cown's message queue depth as a count of
    /// messages processed between token messages falling out of the queue. When
    /// the token message falls out, the `reset_load()` function will push the
    /// upper nibble of the current load into the `load_hist` ring buffer.
    uint8_t _current_load = 0;
    /// Other miscellaneous bits. 7 bits are reserved for future use.
    /// Bit 0 indicates if the token message is in a cown's queue. If zero, a
    /// new token message should be added if the cown runs another message.
    uint8_t _misc = 0;

  public:
    /// Return the count of messages processed since the last token message fell
    /// out of this cown's queue.
    inline uint8_t current_load() const
    {
      return _current_load;
    }

    /// Calculate the total load accumulated for this cown. This value will be
    /// in the range [0, 1215].
    inline uint32_t total_load() const
    {
      // Add the current load to the 4 upper nibbles stored as load history.
      const uint32_t h3 = (uint32_t)bits::extract<15, 12>(_load_hist) << 4;
      const uint32_t h2 = (uint32_t)bits::extract<11, 8>(_load_hist) << 4;
      const uint32_t h1 = (uint32_t)bits::extract<7, 4>(_load_hist) << 4;
      const uint32_t h0 = (uint32_t)bits::extract<3, 0>(_load_hist) << 4;
      return h3 + h2 + h1 + h0 + _current_load;
    }

    /// Return true if this cown is overloaded.
    inline bool overloaded() const
    {
      return total_load() > overload_threshold;
    }

    /// Increment the current load. This increment will become saturated at 255.
    inline void inc_load()
    {
      _current_load += 1;
    }

    /// Reset the current load and store its upper nibble in the load history.
    inline void reset_load()
    {
      _load_hist <<= 4;
      _load_hist |= (_current_load >> 4);
      _current_load = 0;
    }

    /// Artificially overload a cown.
    /// TODO: In the future, this will require a bit to remain overloaded after
    /// the load drops.
    inline void overload()
    {
      _load_hist = (uint16_t)-1;
      _current_load = (uint8_t)-1;
    }

    /// Indicates if the token message is in a cown's queue. If zero, a
    /// new token message should be added if the cown runs another message.
    inline bool has_token()
    {
      return bits::extract<0, 0>(_misc) == 1;
    }

    /// Set whether the cown has a token in its queue.
    inline void set_has_token(bool value)
    {
      _misc = (_misc & 0xFE) | (uint8_t)value;
    }
  };
}
