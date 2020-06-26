// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "ds/bits.h"

namespace verona::rt::backpressure
{
  /**
   * Backpressure bits are contain the following fields:
   * |31          24|23              8|7      0|
   * |   pressure   |   load history  |  load  |
   *
   * The 8-bit pressure field is 0 when muted and in the range [1, 255]
   * otherwise.
   *
   * The 16-bit load history field is a ring buffer with a capacity for 4
   * 4-bit entries.
   *
   * The 8-bit load field tracks an approximation of a cown's message queue
   * depth as a count of messages processed between token messages falling out
   * of the queue. When the token message falls out, the `reset()` function
   * will push the upper nibble of the load into the load history ring buffer.
   */
  using backpressure_t = uint32_t;

  static constexpr uint8_t pressure_shift = 24;
  static constexpr uint32_t pressure_mask = (uint32_t)0xff << pressure_shift;

  static constexpr uint32_t init = (uint32_t)1 << pressure_shift;
  static constexpr size_t overload_threshold = 800;
  static constexpr size_t unmute_threshold = 100;

  /**
   * Return the current pressure value.
   */
  static inline uint8_t pressure(uint32_t bp)
  {
    return (bp & pressure_mask) >> pressure_shift;
  }

  /**
   * Return true if this cown is currently muted and false otherwise.
   */
  static inline uint32_t muted(uint32_t bp)
  {
    return pressure(bp) == 0;
  }

  /**
   * Calculate the total load accumulated for this cown. This value will be in
   * the range [0, 1215].
   */
  static inline uint32_t load(uint32_t bp)
  {
    // Add the current load to the 4 upper nibbles stored as load history. The
    // load history may be more efficiently compressed, but the clang SLP
    // vectorizer optimizes this nicely with AVX2 instructions.
    const uint32_t h3 = (bp & 0xf000'00) >> 16;
    const uint32_t h2 = (bp & 0x0f00'00) >> 12;
    const uint32_t h1 = (bp & 0x00f0'00) >> 8;
    const uint32_t h0 = (bp & 0x000f'00) >> 4;
    return (h3 + h2 + h1 + h0) | (bp & 0xff);
  }

  /**
   * Return if enough pressure has been applied to mute the cown and the cown
   * is not itself under pressure.
   */
  static inline bool should_mute(uint32_t bp)
  {
    return (pressure(bp) == 0xff) && (load(bp) < overload_threshold);
  }

  /**
   * Add pressure without overflowing past 255.
   */
  static inline uint32_t pressure_add(uint32_t bp, uint32_t n)
  {
    auto p = (uint32_t)pressure(bp) + n;
    p = (p > 0xff) ? 0xff : p;
    return (p << pressure_shift) | (bp & ~pressure_mask);
  }

  /**
   * Indicate that the cown has been muted and reset its pressure.
   */
  static inline uint32_t mute(uint32_t bp)
  {
    return bp & ~pressure_mask;
  }

  /**
   * Indicate that the cown is no longer muted and reset its pressure.
   */
  static inline uint32_t unmute(uint32_t bp)
  {
    return (bp & ~pressure_mask) | init;
  }

  /**
   * Increment the current load.
   */
  static inline uint32_t load_inc(uint32_t bp)
  {
    if ((bp & 0xff) < 0xff)
      bp++;
    return bp;
  }

  /**
   * Distribute load from first participant of a multimessage. This is done
   * because the other participants in this multimessage are likely to have
   * their load reset before they are acquired for the multimessage.
   */
  static inline uint32_t distribute_load(uint32_t bp, uint32_t bp0)
  {
    return bp | (bp0 & 0xff);
  }

  /**
   * Reset the current load to zero and store the upper nibble into the load
   * history buffer.
   */
  static inline uint32_t load_reset(uint32_t bp)
  {
    const uint32_t hist = (bp & 0x00'0fff'f0) << 4;
    return (bp & pressure_mask) | hist;
  }
}
