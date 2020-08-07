// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

/**
 * ## Introduction
 *
 * The primary goal of backpressure is to prevent runaway growth of a cown's
 * message queue that might eventually result in a Verona application running
 * out of memory.
 *
 * The following expects the reader to be familiar with scheduler threads,
 * work stealing, fairness, cown reference counting, multi-message acquisition,
 * and actions.
 *
 * If a cown receives messages more quickly than it is able to process them, the
 * queue will start to grow in an unbounded fashion. Cowns that aren't able to
 * keep up with their arriving messages are marked as "overloaded". Once a cown
 * is marked as overloaded, cowns sending messages to it may be temporarily
 * descheduled ("muted"). Descheduling cowns that are sending to an overloaded
 * cown helps alleviate pressure on the it by:
 *   - Temporarily removing a source of incoming messges
 *   - Allowing more scheduler time to be spent on the overload cown rather than
 *     the descheduled ones.
 *
 * Similarly, non-overloaded cowns sending to muted cowns may also be muted to
 * prevent runaway queue growth while the muted cown in descheduled.
 *
 * Muted cowns will be rescheduled after the receiver that triggered their
 * muting (the "mutor") is no longer muted or overloaded.
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
 * ## Backpressure Response
 *
 * The backpressure system is designed to respond to an overloaded cown by
 * muting cowns that contribute to its load. However, some cowns may be
 * temporarily unmutable in order to ensure that overload cowns make progress.
 * In this system a cown may transition between many states as they interact
 * with cowns contributing to the backpressure response, as shown in the
 * following diagram:
 *
 *      +--------+     +-----------+
 *  +-->|        |---->|           |---+
 *      | Normal |     | Unmutable |   |
 *      |        |<----|           |<--+
 *      +--------+     +-----------+
 *        |   ^         ^
 *        |   |         |
 *        v   |         |
 *      +-------+       |
 *      |       |-------+
 *      | Muted |
 *      |       |
 *      +-------+
 *
 * State Properties:
 * - Normal: The inital state of all cowns.
 * - Muted: A muted cown may not be scheduled or deallocated.
 * - Unmutable: An nnmutable cown may not be muted.
 *
 * State Transition Rules:
 * - Cowns sending to overload/muted cowns may be muted.
 * - A cown is unmutable so long as a message from an overloaded cown is in its
 *   queue.
 *
 * ## Scanning Messages
 *
 * All messages from one set of sending cowns to a set of receiving cowns are
 * scanned to determine if the senders should be muted. The senders will be
 * muted once their behaviour completes if all of the following are true:
 *   1. Any of the receivers are either overloaded or muted.
 *   2. The receiver set does not contain any of the senders.
 *   3. None of the senders are overloaded.
 *
 * If all of the above conditions are met, the first receiver that is either
 * overloaded or muted is identified as the mutor for the senders. First in this
 * case means the least cown, determined by the multi-message cown order. All
 * senders that are in the "normal" (therefore mutable) state will be muted.
 *
 * ## Tracking Muted Cowns
 *
 * Once a scheduler thread completes a message behaviour that would result in
 * muting the cowns running the messsage, the cowns are marked as muted,
 * acquired by the scheduler thread, and placed in a "mute map" on the scheduler
 * thread. The mute map is a mapping from mutor => mute set, where "mute set"
 * refers to the set of cowns muted by a mutor.
 *
 * A mutor may exist as a key in mutiple mute maps, since cowns may continue
 * sending messages to it on other threads and then be subsequently muted by
 * that thread on which the senders were running. A muted cown may also exist in
 * multiple mute sets across scheduler threads.
 *
 * A muted cown may not be scheduled or collected until they are marked as no
 * longer muted and recheduled ("unmuted").
 *
 * Each scheduler thread scans the mutors of its mute map on each iteration of
 * the run loop. If any mutor is not muted and is not overloaded, all cowns in
 * the corresponding mute set are unmuted and the entry is removed from the mute
 * map.
 *
 * ## Limitations
 *
 * Prediction:
 * The backpressure system does not attempt to predict the consequences of a
 * cown sending a message before the first message is sent. Backpressure is only
 * applied to senders after they have already contributed to overloading a set
 * of cowns. So the first message sent by a cown can always result in an
 * unchecked storm of messages to some set of receivers. In general, such
 * prediction is considered prohibitively expensive.
 *
 * Deadlock Prevention:
 * The backpressure system is designed to achieve its goals while maintaining
 * properties of the Verona runtime, such as causal message order and deadlock
 * freedom. Deadlock freedom limits the backpressure system because a
 * multi-message including an overload cown may be indefinitely postponed if any
 * other participants are muted. So, in order to prevent deadlock, the cowns
 * that an overload cown interacts with (i.e. sends messages to or participates
 * in multi-messages with) are unmutable so long as a message from an overloaded
 * cown exists in its message queue.
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
    // Load at which cown is overloaded.
    static constexpr uint32_t overload_threshold = 800;
    // Load at which a mutor triggers unmuting, if not muted.
    static constexpr uint32_t unmute_threshold = 100;

    /**
     * Tracks an approximation of a cown's message queue depth as a count of
     * messages processed between token messages falling out of the queue. When
     * the token message falls out, the `reset_load()` function will push the
     * upper nibble of the current load into the load history ring buffer.
     */
    uint32_t _current_load : 8;
    /**
     * Ring buffer with a capacity for 4 4-bit entries.
     */
    uint32_t _load_hist : 16;
    /**
     * Reserved for future use.
     */
    uint32_t : 5;
    /**
     * Indicates if the token message is in a cown's queue. If zero, a new token
     * message should be added if the cown runs another message.
     */
    uint32_t _has_token : 1;
    /**
     * Indicates if a cown is currently muted (1), unmutable (2), or
     * unmutable-dirty (3).
     */
    uint32_t _response_state : 2;

  public:
    Backpressure() noexcept
    : _current_load(0), _load_hist(0), _has_token(0), _response_state(0)
    {}

    bool operator==(Backpressure& other) const
    {
      return *(uint32_t*)this == *(uint32_t*)&other;
    }

    /**
     * Return true if this cown is muted.
     */
    inline bool muted() const
    {
      return _response_state == 1;
    }

    /**
     * Return true if this cown is unmutable or unmutable-dirty.
     */
    inline bool unmutable() const
    {
      return (_response_state & 0b10) != 0;
    }

    /**
     * Return true if this cown may become normal when the next token message is
     * reached.
     */
    inline bool unmutable_dirty() const
    {
      return _response_state == 3;
    }

    /**
     * Return true if the token message is in this cown's queue.
     */
    inline bool has_token() const
    {
      return _has_token == 1;
    }

    /**
     * Return the count of messages processed since the last token message fell
     * out of this cown's queue.
     */
    inline uint8_t current_load() const
    {
      return _current_load;
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
      const uint32_t h3 = bits::extract<15, 12>(_load_hist) << 4;
      const uint32_t h2 = bits::extract<11, 8>(_load_hist) << 4;
      const uint32_t h1 = bits::extract<7, 4>(_load_hist) << 4;
      const uint32_t h0 = bits::extract<3, 0>(_load_hist) << 4;
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
     * Return true if mutable senders should be muted.
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
     * Mark this cown as normal.
     */
    inline void set_state_normal()
    {
      assert(muted() || unmutable_dirty());
      _response_state = 0;
    }

    /**
     * Mark this cown as muted.
     */
    inline void set_state_muted()
    {
      assert(_response_state == 0);
      _response_state = 1;
    }

    /**
     * Mark this cown as unmutable.
     */
    inline void set_state_unmutable()
    {
      _response_state = 2;
    }

    /**
     * Mark this cown as unmutable-dirty.
     */
    inline void set_state_unmutable_dirty()
    {
      assert(unmutable());
      _response_state = 3;
    }

    /**
     * Mark that this cown has a token message in its queue.
     */
    inline void set_has_token()
    {
      _has_token = 1;
    }

    /**
     * Mark that this cown has removed the token message from its queue.
     */
    inline void unset_has_token()
    {
      _has_token = 0;
    }

    /**
     * Increment the current load. This increment will become saturated at 255.
     */
    inline void inc_load()
    {
      _current_load += 1;
    }

    /**
     * Reset the current load and store its upper nibble in the load history.
     */
    inline void reset_load()
    {
      _load_hist <<= 4;
      _load_hist |= (_current_load >> 4);
      _current_load = 0;
    }
  };

  static_assert((sizeof(Backpressure) == 4) && (alignof(Backpressure) == 4));
  static_assert(
    (sizeof(Backpressure) == sizeof(std::atomic<Backpressure>)) &&
    (alignof(Backpressure) == alignof(std::atomic<Backpressure>)));
}
