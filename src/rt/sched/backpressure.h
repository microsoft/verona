// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

/**
 * ## Introduction
 *
 * The primary goal of backpressure is to prevent runaway growth of a cown's
 * message queue that might eventually result in a Verona application running
 * out of memory. Cowns that receive messages at a higher rate than they are
 * able to process are considered "overloaded" and may lead to this unbounded
 * queue growth. Once an overloaded cown is identified, cowns sending messages
 * to it will be temporarily descheduled ("muted") so that the overloaded cown
 * can work through its queue. Similarly, cowns sending to muted receivers will
 * also be muted so that cowns do not themselves become overloaded as a
 * consequence of being muted. Muted cowns will be rescheduled after the
 * receiver that triggered their muting (the "mutor") is no longer muted or
 * overloaded.
 *
 * ## Detecting Overloaded Cowns
 *
 * An overloaded cown is one that experiences message queue growth as it is
 * processing messages. Since it would be prohibitively expensive to calculate
 * the length of a message queue directly, a count of messages removed from the
 * queue over some period is used as an approximation for message queue size.
 * Like scheduler fairness, the period is determined using a token message that
 * will occasionally fall out of the queue and then be inserted to begin the
 * next period. The "load" of a cown at any time is the accumulation of messages
 * processed over multiple periods.
 *
 * ## Muting Cowns
 *
 * All messages from one set of sending cowns to a set of receiving cowns are
 * scanned to determine if the senders should be muted. The senders will be
 * muted once their action completes if all of the following are true:
 *   1. The receiver set does not contain any of the senders.
 *   2. No sender is overloaded.
 *   3. Any of the receivers are either overloaded or muted.
 *
 * A message is given "priority" if either condition 1 or 2 are false. A
 * priority message will result in any muted receivers being unmuted so that the
 * senders may make progress sooner.
 *
 * If all of the above conditions are met, the first receiver that is either
 * overloaded or muted is identified as the mutor for the senders. First in this
 * case means the least cown, determined by the mutlimessage cown order.
 *
 * ## Tracking Muted Cowns
 *
 * Once a scheduler thread completes a message action that would result in
 * muting the cowns running the messsage, the cowns are acquired by the
 * scheduler thread and placed in a "mute map" on the scheduler thread. The mute
 * map is a mapping from mutor => mute set, where "mute set" refers to the set
 * of cowns muted by a mutor.
 *
 * A mutor may exist as a key in mutiple mute maps, since cowns may continue
 * sending messages to it on other threads and then be subsequently muted by
 * that thread on which the senders were running. However, a muted cown must
 * only exist in a mute set on the scheduler thread that muted it.
 *
 * A muted cown may not be scheduled or collected until they are removed from
 * their scheduler thread's mute map and rescheduled by that thread ("unmuted").
 *
 * Each scheduler thread scans the mutors of its mute map on each iteration of
 * the run loop. If any mutor is not muted and is not overloaded, all cowns in
 * the corresponding mute set are unmuted and the entry is removed from the mute
 * map.
 */

#include "ds/mpscq.h"

namespace verona::rt
{
  /**
   * Tracks information related to backpressure for a cown. This class may only
   * be modified by the scheduler thread that is either running the cown or is
   * tracking the muted cown in its mute map.
   */
  class Backpressure
  {
    static constexpr uint32_t current_load_mask = 0x00'0000'ff;
    static constexpr uint32_t load_hist_mask = 0x00'ffff'00;
    static constexpr uint32_t muted_mask = 0x80'0000'00;
    static constexpr uint32_t needs_token_mask = 0x40'0000'00;

    static constexpr uint32_t overload_threshold = 800;
    static constexpr uint32_t unmute_threshold = 100;

    /**
     * Backpressure bits contain the following fields:
     *   [31:31] muted
     *   [30:30] needs_token
     *   [29:24] (unused)
     *   [23:08] load_history
     *   [07:00] current_load
     *
     * The `muted` bit indicates if a cown is currently muted.
     *
     * The `needs_token` bit indicates if the token message is not in a cown's
     * queue and that a new token message should be added if it runs another
     * message.
     *
     * The 16-bit `load_history` field is a ring buffer with a capacity for 4
     * 4-bit entries.
     *
     * The 8-bit load field tracks an approximation of a cown's message queue
     * depth as a count of messages processed between token messages falling out
     * of the queue. When the token message falls out, the `reset_load()`
     * function will push the upper nibble of the current load into the load
     * history ring buffer.
     */
    uint32_t bits = 0 | needs_token_mask;

  public:
    /**
     * Return true if this cown is muted.
     */
    inline bool muted() const
    {
      return (bits & muted_mask) != 0;
    }

    /**
     * Return true if the token message is not in this cown's queue and a new
     * token message should be added if it runs another message.
     */
    inline bool needs_token() const
    {
      return (bits & needs_token_mask) != 0;
    }

    /**
     * Return the count of messages processed since the last token message fell
     * out of this cown's queue.
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
      // load history could be more efficiently compressed, but the clang SLP
      // vectorizer optimizes this nicely with AVX2 instructions.
      const uint32_t h3 = (bits & 0x00'f000'00) >> 16;
      const uint32_t h2 = (bits & 0x00'0f00'00) >> 12;
      const uint32_t h1 = (bits & 0x00'00f0'00) >> 8;
      const uint32_t h0 = (bits & 0x00'000f'00) >> 4;
      return (h3 + h2 + h1 + h0) | current_load();
    }

    /**
     * Return true if this cown is overloaded.
     */
    inline bool overloaded() const
    {
      return total_load() > overload_threshold;
    }

    /**
     * If this cown is recieving a message without priority, return true if the
     * senders should be muted.
     */
    inline bool triggers_muting() const
    {
      return muted() || overloaded();
    }

    /**
     * If this cown is a key in a mute map, return true if the correspinding
     * mute set should be unmuted.
     */
    inline bool triggers_unmuting() const
    {
      return !muted() && (total_load() < unmute_threshold);
    }

    /**
     * Mark this cown as muted.
     */
    inline void set_muted()
    {
      assert(!muted());
      bits |= muted_mask;
    }

    /**
     * Mark this cown as not muted.
     */
    inline void unset_muted()
    {
      assert(muted());
      bits &= ~muted_mask;
    }

    /**
     * Mark that this cown has a token message in its queue.
     */
    inline void set_needs_token()
    {
      bits &= ~needs_token_mask;
    }

    /**
     * Mark that this cown has removed the token message from its queue.
     */
    inline void unset_needs_token()
    {
      bits |= needs_token_mask;
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
     * Reset the current load and store its upper nibble in the load history.
     */
    inline void reset_load()
    {
      const uint32_t hist = (bits & 0x00'0fff'f0) << 4;
      bits = (bits & 0xff'0000'00) | hist;
    }
  };

  template<typename T>
  struct UnmuteMessage
  {
    std::atomic<UnmuteMessage*> next;
    T* cown;

    UnmuteMessage(T* cown_) : cown(cown_) {}

    size_t size() const
    {
      return sizeof(UnmuteMessage<T>);
    }
  };
}
