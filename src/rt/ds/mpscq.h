// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include <atomic>
#include <snmalloc.h>

namespace verona::rt
{
  // Forward reference for systematic testing
  static void yield();

  /**
   * MPSCQ - Multi Producer Single Consumer Queue.
   *
   * This queue allows multiple threads to insert data into the queue. Removing
   * elements is not thread safe, so can only be done by a single scheduler
   * thread.
   *
   * Elements are enqueued (added) to the `back` of the queue and dequeued
   * (removed) from the `front`.  The queue is considered empty when `back`
   * and `front` contain the same value.
   *
   * The queue always contains one stub element.  This removes some branching in
   * the implementation.
   *
   * The queue supports several internal states to enable schedulers to manage
   * the ownership of the queue.
   *
   *    None
   *      This is the standard state of the queue. It is owned by something,
   *      which is expected to process the messages.
   *
   *    Sleeping
   *      This means the queue does not contain any messages.  Any enqueue to
   *      the message queue will be informed that the queue was "sleeping". Only
   *      a single enqueuer will be informed that it woke the queue up, so this
   *      can be used to represent taking ownership of the queue.
   *
   *      Note that the queue can be empty and not asleep.
   *
   *    Delay
   *      This state means prevents going to sleep immediately. The next call to
   *      mark_sleeping is guaranteed to fail, either because the queue is still
   *      in this state, or a new message has been enqueued.  This is used to
   *      prevent the queue going to sleep, directly after a reschedule from the
   *      runtime.
   *
   *    Notify
   *      The queue supports a single consolidated message type that has no
   *      payload and does not require any allocation, a notificaton.  If the
   *      queue receives multiple calls to "notify" it may consolidate them into
   *      a single call. This supports zero allocation notifications in the
   *      runtime.
   **/

  template<class T>
  class MPSCQ
  {
  private:
    static_assert(
      std::is_same<decltype(((T*)nullptr)->next), std::atomic<T*>>::value,
      "T->next must be a std::atomic<T*>");

    // Embedding state into last two bits.
    enum STATE
    {
      NONE = 0x0,
      SLEEPING = 0x1,
      DELAY = 0x2,
      NOTIFY = 0x3,
      STATES = 0x3,
    };

    static constexpr uintptr_t MASK = ~static_cast<uintptr_t>(STATES);

    std::atomic<T*> back;
    T* front;

    inline static bool has_state(T* p, STATE f)
    {
      return ((uintptr_t)p & STATES) == f;
    }

    inline static T* set_state(T* p, STATE f)
    {
      assert(is_clear(p));
      return (T*)((uintptr_t)p | f);
    }

    inline static bool is_clear(T* p)
    {
      return clear_state(p) == p;
    }

    inline static STATE get_state(T* p)
    {
      return static_cast<STATE>((uintptr_t)p & STATES);
    }

    static T* clear_state(T* p)
    {
      return (T*)((uintptr_t)p & MASK);
    }

  public:
    void invariant()
    {
#ifndef NDEBUG
      assert(back != nullptr);
      assert(front != nullptr);
#endif
    }

    void init(T* stub)
    {
      stub->next.store(nullptr, std::memory_order_relaxed);
      front = stub;

      stub = set_state(stub, SLEEPING);

      back.store(stub, std::memory_order_relaxed);
      invariant();
    }

    T* destroy()
    {
      T* fnt = front;
      back.store(nullptr, std::memory_order_relaxed);
      front = nullptr;
      return fnt;
    }

    T* peek_back()
    {
      return clear_state(back.load(std::memory_order_relaxed));
    }

    inline bool is_sleeping()
    {
      T* bk = back.load(std::memory_order_relaxed);

      return has_state(bk, SLEEPING);
    }

    /**
     * Enqueues (inserts) a message into the queue.
     *
     * Returns true if the queue was sleeping when the message was added.
     **/
    bool enqueue(T* t)
    {
      assert(is_clear(t));

      invariant();
      t->next.store(nullptr, std::memory_order_relaxed);
      std::atomic_thread_fence(std::memory_order_release);
      T* prev = back.exchange(t, std::memory_order_relaxed);
      bool was_sleeping;

      yield();

      // Pass on the notify info if set
      if (has_state(prev, NOTIFY))
      {
        t = set_state(t, NOTIFY);
      }

      was_sleeping = has_state(prev, SLEEPING);
      prev = clear_state(prev);

      prev->next.store(t, std::memory_order_relaxed);
      return was_sleeping;
    }

    /**
     * Dequeues (removes) an element from the queue
     *
     * Returns nullptr if the queue is empty.
     *
     * If it returns a message, will delete the previous message.
     *
     * Messages are deallocated after the next message is dequeued. This ensures
     * that there is always a message in the queue.
     **/
    T* dequeue(snmalloc::Alloc* alloc, bool& notify)
    {
      // Returns the next message. If the next message
      // is not null, the front message is freed.
      invariant();
      T* fnt = front;
      assert(is_clear(fnt));
      T* next = fnt->next.load(std::memory_order_relaxed);

      if (next == nullptr)
      {
        return nullptr;
      }

      front = clear_state(next);

      assert(front);
      std::atomic_thread_fence(std::memory_order_acquire);

      alloc->dealloc(fnt, fnt->size());
      invariant();

      if (has_state(next, NOTIFY))
      {
        next = clear_state(next);
        notify = true;
      }

      return next;
    }

    /**
     * Used to find the first element in the queue. Only safe to use in the
     * consumer.
     **/
    T* peek()
    {
      return clear_state(front->next.load(std::memory_order_relaxed));
    }

    /**
     * Used to set the NOTIFY state on the queue. Returns true if the queue
     * was previously SLEEPING.
     *
     *  mark_notify; mark_sleeping;
     *
     * The mark_sleeping will have its NOTIFY status set.
     *
     *   mark_notify; enqueue; enqueue; dequeue;
     *
     * The dequeue call will have its NOTIFY status set.
     *
     * Note that the calls are consolidated:
     *
     *   mark_notify; mark_notify; enqueue; enqueue; dequeue; dequeue;
     *
     * will only result in the first dequeue having the notify parameter set.
     *
     * State transition:
     *   NONE     -> NOTIFY;  return false
     *   SLEEPING -> NOTIFY;  return true
     *   DELAY    -> NOTIFY;  return false
     *   NOTIFY   -> NOTIFY;  return false
     * Scheduling is required when the queue was SLEEPING, but not other states.
     */
    bool mark_notify()
    {
      auto bk = back.load(std::memory_order_relaxed);
      auto was_sleeping = false;

      while (true)
      {
        if (has_state(bk, NOTIFY))
        {
          break;
        }

        auto notify = set_state(clear_state(bk), NOTIFY);

        if (back.compare_exchange_strong(bk, notify, std::memory_order_release))
        {
          was_sleeping = has_state(bk, SLEEPING);
          break;
        }
      }

      return was_sleeping;
    }

    /**
     * Attempts to set the queue into a SLEEPING state.  Will only succeed if
     * the queue is empty and in the NONE state, and wake has not been called
     * since the queue became empty. Returns true if the queue was successfully
     * set to SLEEPING.
     *
     * Note that for a sequential call sequence
     *
     *    wake; mark_sleeping; mark_sleeping;
     *
     * the second call to mark_sleeping will succeed.
     *
     * Similarly
     *
     *    wake; enqueue; dequeue; mark_sleeping;
     *
     * the call to mark_sleeping will succeed assuming the queue is empty.
     *
     * The notify parameter will be set if the notification has not yet been
     * observed by a previous mark_sleeping.
     *
     * State transition (for a non-empty queue):
     *   NONE     -> NONE;      return false
     *   SLEEPING -> ABORT;     invalid input
     *   DELAY    -> NONE;      return false
     *   NOTIFY   -> NONE;      return false, and set notify argument to true
     *
     * State transition (for an empty queue):
     *   NONE     -> SLEEPING;  return true
     *   else     -> ABORT;     invalid input
     * Only safe to call from the consumer.
     */
    bool mark_sleeping(bool& notify)
    {
      T* fnt = front;
      T* bk = back.load(std::memory_order_relaxed);

      if (bk != fnt)
      {
        switch (get_state(bk))
        {
          case NONE:
            return false;
          case SLEEPING:
            // Only the consumer can call `mark_sleeping`. The consumer should
            // not call `mark_sleeping` is the queue is SLEEPING.
            abort();
          case DELAY:
          {
            T* clear = clear_state(bk);
            back.compare_exchange_strong(bk, clear, std::memory_order_release);
            return false;
          }
          case NOTIFY:
          {
            notify = true;
            T* clear = clear_state(bk);
            back.compare_exchange_strong(bk, clear, std::memory_order_release);
            return false;
          }

          default:
            abort();
        }
      }

      // note: set_state asserts that fnt is in the NONE state
      bk = set_state(fnt, SLEEPING);
      return back.compare_exchange_strong(fnt, bk, std::memory_order_release);
    }

    /**
     * Prevents a single subsequent call to mark_sleeping from suceeding unless
     * a new message is enqueued and dequeued. Returns true if the queue was
     * previously SLEEPING. Safe to call from a producer.
     *
     * State transition:
     *   NONE     -> DELAY|Other;  return false
     *   SLEEPING -> NONE;         return true
     *   DELAY    -> DELAY;        return false
     *   NOTIFY   -> NOTIFY;       return false
     * (`Other` means that another thread beats us in CAS so we don't know for
     * sure what the state is now.)
     */
    bool wake()
    {
      T* bk = back.load(std::memory_order_relaxed);
      T* clear = clear_state(bk);
      T* delay = set_state(clear, DELAY);

      if (bk == delay)
        return false;

      if (has_state(bk, NOTIFY))
      {
        // Preserve NOTIFY bit
        return false;
      }

      if (
        (bk == clear) &&
        back.compare_exchange_strong(bk, delay, std::memory_order_release))
      {
        return false;
      }

      T* sleeping = set_state(clear, SLEEPING);
      return back.compare_exchange_strong(
        sleeping, clear, std::memory_order_release);
    }
  };
} // namespace verona::rt
