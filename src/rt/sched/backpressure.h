// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

/** TODO: cleanup these notes
 * ## Muting
 * A cown is muted after it runs a message where it has sent a message to an
 * overloaded/muted cown and no sender participating in the message is
 * overloaded. Once a cown is muted, the scheduler thread running the cown
 * becomes responsible for eventually unmuting it.
 *
 * ## Unmuting
 * - The cown that resulted in the muting (the key in the mute map) is no longer
 * overloaded/muted. In this case, an unmute message is sent to each scheduler
 * thread to unmute all muted cowns with the unmuted cown as its key. The entry
 * in the mute map is removed and all unmuted cowns are removed from the mute
 * set.
 * - An overloaded cown is sending a message to the muted cown. In this case,
 * the muted cown is unmuted. The responsible scheduler thread is expected to
 * not unmute cowns that have already been unmuted and to re-mute cowns that
 * have been unmuted. (exchange on muted bit?)
 *
 *
 * Cown Backpressure States:
 *
 *     +------------+    +------------+
 *  +--|            |--->|            |
 *  |  | Normal     |    | Muting     |
 *  +->|            |<-+-|            |
 *     +------------+  ^ +------------+
 *          | ^        |      | ^
 *          v |        |      v |
 *     +------------+  | +------------+
 *     |            |  | |            |
 *     | Overloaded |  +-| Muted      |
 *     |            |    |            |
 *     +------------+    +------------+
 *
 */

namespace verona::rt
{
  class Backpressure
  {
    static constexpr uint32_t current_load_mask = 0x00'0000'ff;
    static constexpr uint32_t load_hist_mask = 0x00'ffff'00;
    static constexpr uint32_t muted_mask = 0x80'0000'00;

    /**
     * Backpressure bits are contain the following fields:
     * |31       31|30   24|23              8|7              0|
     * |   muted   |   ?   |   load history  |  current load  |
     *
     * TODO
     *
     * The 16-bit load history field is a ring buffer with a capacity for 4
     * 4-bit entries.
     *
     * The 8-bit load field tracks an approximation of a cown's message queue
     * depth as a count of messages processed between token messages falling out
     * of the queue. When the token message falls out, the `reset_load()`
     * function will push the upper nibble of the current load into the load
     * history ring buffer.
     */
    uint32_t bits = 0;

  public:
    /**
     * TODO
     */
    inline bool muted() const
    {
      return (bits & muted_mask) != 0;
    }

    /**
     * TODO
     */
    inline uint8_t current_load() const
    {
      return bits & current_load_mask;
    }

    /**
     * Calculate the total load accumulated for this cown. This value will be
     * in the range [0, 1215].
     */
    inline uint32_t total_load() const
    {
      // Add the current load to the 4 upper nibbles stored as load history. The
      // load history may be more efficiently compressed, but the clang SLP
      // vectorizer optimizes this nicely with AVX2 instructions.
      const uint32_t h3 = (bits & 0x00'f000'00) >> 16;
      const uint32_t h2 = (bits & 0x00'0f00'00) >> 12;
      const uint32_t h1 = (bits & 0x00'00f0'00) >> 8;
      const uint32_t h0 = (bits & 0x00'000f'00) >> 4;
      return (h3 + h2 + h1 + h0) | current_load();
    }

    /**
     * TODO. This value will be in the range [0, 151].
     */
    inline uint8_t shifted_total_load() const
    {
      return total_load() >> 3;
    }

    /**
     * Increment the current load. This increment will become saturated at 255.
     */
    inline void inc_load()
    {
      if (current_load() < 0xff)
        bits++;
    }

    /**
     * Reset the current load to zero and store the upper nibble into the load
     * history buffer.
     */
    inline void reset_load()
    {
      const uint32_t hist = (bits & 0x00'0fff'f0) << 4;
      bits = (bits & ~current_load_mask) | hist;
    }
  };

  template<typename T>
  class BackpressureMessage
  {
    friend MPSCQ<BackpressureMessage<T>>;

    std::atomic<BackpressureMessage<T>*> next;

    BackpressureMessage(T* key_, T* cown_) : key(key_), cown(cown_) {}

  public:
    T* key;
    T* cown;

    static BackpressureMessage<T> mute(T* key, T* cown)
    {
      return BackpressureMessage<T>(key, cown);
    }

    static BackpressureMessage<T> unmute(T* cown)
    {
      return BackpressureMessage<T>(nullptr, cown);
    }
  };

  // MultiMessageBody* body;
  //   friend verona::rt::MPSCQ<MultiMessage>;
  //   friend class Cown;

  //   std::atomic<MultiMessage*> next;
}
