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
 * is marked as overloaded, cowns sending messages to it will be temporarily
 * descheduled ("muted"). Descheduling cowns that are sending to an overloaded
 * cown helps alleviate pressure on the it by:
 *   - Temporarily removing a source of incoming messges
 *   - Allowing more scheduler time to be spent on the overload cown rather than
 *     the descheduled ones.
 *
 * Similarly, cowns sending to cowns that have been muted will also be muted to
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
 * ## Muting Cowns
 *
 * All messages from one set of sending cowns to a set of receiving cowns are
 * scanned to determine if the senders should be muted. The senders will be
 * muted once their action completes if all of the following are true:
 *   1. Any of the receivers are either overloaded or muted.
 *   2. The receiver set does not contain any of the senders.
 *   3. No sender is overloaded.
 *
 * A message is given "priority" if either condition 2 or 3 are false, i.e., if
 * any receiver is overloaded or is sending to itself. A priority message will
 * result in any muted receivers being unmuted so that the senders may make
 * progress sooner.
 *
 * If all of the above conditions are met, the first receiver that is either
 * overloaded or muted is identified as the mutor for the senders. First in this
 * case means the least cown, determined by the multimessage cown order.
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
 *
 * ## Limitations
 *
 * Fan-in/Fan-out:
 * The backpressure system does not attempt to predict the consequences of a
 * cown sending a message before the first message is sent. Backpressure is only
 * applied to senders after they have already contributed to overloading a set
 * of cowns. So the first message sent by a cown can always result in an
 * unchecked storm of messages to some set of receivers. In general, such
 * prediction is considered prohibitively expensive.
 *
 * Frequent Unmute for Priority:
 * We cannot ensure that a cown unmuted by a priority message will only handle
 * that message required for progress before becoming muted once again. The
 * perf-backpressure3 test shows a scenario where runaway receiver queue growth
 * could be prevented. However, a general solution would require reordering the
 * messages processed by the producer, which would break causal message
 * ordering.
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
    uint32_t : 6;
    /**
     * Indicates if the token message is not in a cown's queue and that a new
     * token message should be added if it runs another message.
     */
    uint32_t _needs_token : 1;
    /**
     * Indicates if a cown is currently muted.
     */
    uint32_t _muted : 1;

  public:
    Backpressure() : _current_load(0), _load_hist(0), _needs_token(1), _muted(0)
    {}

    /**
     * Return true if this cown is muted.
     */
    inline bool muted() const
    {
      return _muted != 0;
    }

    /**
     * Return true if the token message is not in this cown's queue and a new
     * token message should be added if it runs another message.
     */
    inline bool needs_token() const
    {
      return _needs_token != 0;
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
      _muted = 1;
    }

    /**
     * Mark this cown as not muted.
     */
    inline void unset_muted()
    {
      assert(muted());
      _muted = 0;
    }

    /**
     * Mark that this cown has a token message in its queue.
     */
    inline void set_needs_token()
    {
      _needs_token = 1;
    }

    /**
     * Mark that this cown has removed the token message from its queue.
     */
    inline void unset_needs_token()
    {
      _needs_token = 0;
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
      _load_hist &= 0x0fff;
      _load_hist <<= 4;
      _load_hist |= (_current_load >> 4);
      _current_load = 0;
    }
  };

  static_assert((sizeof(Backpressure) == 4) && (alignof(Backpressure) == 4));
  static_assert(
    (sizeof(Backpressure) == sizeof(std::atomic<Backpressure>)) &&
    (alignof(Backpressure) == alignof(std::atomic<Backpressure>)));

  template<typename T>
  struct UnmuteMessage
  {
    std::atomic<UnmuteMessage*> next{nullptr};
    T* cown;

    UnmuteMessage(T* cown_) : cown(cown_) {}

    size_t size() const
    {
      return sizeof(UnmuteMessage<T>);
    }
  };
}
