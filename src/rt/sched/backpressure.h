// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "ds/bits.h"

namespace verona::rt
{
  class Backpressure
  {
    static constexpr uint8_t pressure_shift = 24;
    static constexpr uint32_t pressure_mask = (uint32_t)0xff << pressure_shift;
    static constexpr uint32_t init = (uint32_t)1 << pressure_shift;

    /**
     * Backpressure bits are contain the following fields:
     * |31          24|23              8|7      0|
     * |   pressure   |   load history  |  load  |
     *
     * The 8-bit pressure field is 0 when muted and in the range [1, 255]
     * otherwise. A non-zero value is collected by incrementing the caunter by a
     * value proportional to the load on a receiving cown when this cown sends a
     * message to it. A maximum value of 255 will eventually result in this cown
     * being muted.
     *
     * The 16-bit load history field is a ring buffer with a capacity for 4
     * 4-bit entries.
     *
     * The 8-bit load field tracks an approximation of a cown's message queue
     * depth as a count of messages processed between token messages falling out
     * of the queue. When the token message falls out, the `reset()` function
     * will push the upper nibble of the load into the load history ring buffer.
     */
    uint32_t bits = init;

  public:
    static constexpr size_t overload_threshold = 800;
    static constexpr size_t unmute_threshold = 100;

    /**
     * Return the current pressure value.
     */
    inline uint8_t pressure() const
    {
      return (bits & pressure_mask) >> pressure_shift;
    }

    /**
     * Return the current load value.
     */
    inline uint8_t current_load() const
    {
      return bits & 0xff;
    }

    /**
     * Calculate the total load accumulated for this cown. This value will be in
     * the range [0, 1215].
     */
    inline uint32_t total_load() const
    {
      // Add the current load to the 4 upper nibbles stored as load history. The
      // load history may be more efficiently compressed, but the clang SLP
      // vectorizer optimizes this nicely with AVX2 instructions.
      const uint32_t h3 = (bits & 0xf000'00) >> 16;
      const uint32_t h2 = (bits & 0x0f00'00) >> 12;
      const uint32_t h1 = (bits & 0x00f0'00) >> 8;
      const uint32_t h0 = (bits & 0x000f'00) >> 4;
      return (h3 + h2 + h1 + h0) | (bits & 0xff);
    }

    /**
     * Return true if this cown is currently muted and false otherwise.
     */
    inline bool muted() const
    {
      return pressure() == 0;
    }

    /**
     * Return true if enough pressure has been applied to mute the cown and the
     * cown is not itself under pressure.
     */
    inline bool should_mute() const
    {
      return (pressure() == 0xff) && (total_load() < overload_threshold);
    }

    /**
     * Indicate that the cown has been muted and reset its pressure.
     */
    inline void mute()
    {
      bits &= ~pressure_mask;
    }

    /**
     * Indicate that the cown is no longer muted and reset its pressure.
     */
    inline void unmute()
    {
      bits = (bits & ~pressure_mask) | init;
    }

    /**
     * Add pressure without overflowing past 255.
     */
    inline void pressure_add(uint32_t n)
    {
      auto p = (uint32_t)pressure() + n;
      p = (p > 0xff) ? 0xff : p;
      bits = (p << pressure_shift) | (bits & ~pressure_mask);
    }

    /**
     * Increment the current load.
     */
    inline void load_inc()
    {
      if (current_load() < 0xff)
        bits++;
    }

    /**
     * Reset the current load to zero and store the upper nibble into the load
     * history buffer.
     */
    inline void load_reset()
    {
      const uint32_t hist = (bits & 0x00'0fff'f0) << 4;
      bits = (bits & pressure_mask) | hist;
    }

    /**
     * Distribute load from first participant of a multimessage.
     */
    inline void distribute_load(const Backpressure& bp0)
    {
      // This operation makes no change when applied to the first cown of a
      // multimessage and otherwise replaces the current load count with that of
      // the first cown. We expect that all participants acquired after the
      // first are likely to have had their load counters reset to 1 when
      // acquired for the message.
      bits |= (uint32_t)bp0.current_load();
    }
  };
}
