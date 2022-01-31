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
   * (removed) from the `front`.  
   *
   * The queue may contain one stub element.  This removes some branching in
   * the implementation.
   *
   * The queue is considered empty when 
   *    1. `back` and `front` contain the same value, and the queue contains a stub element, or
   *    2. `front` is nullptr, and `back` is `&front`.
   * 
   * The queue supports several internal states to enable schedulers to manage
   * the ownership of the queue.
   *
   *    None
   *      This is the standard state of the queue. It is owned by something,
   *      which is expected to process the messages.
   *
   *    Stub
   *      This is used to indicate the initial message in the queue is a stub.
   *      Hence, the first real message is `front->next` rather than `front`.
   *      Stub is indicated by the setting the stub bit on the `front`.
   *
   *    Sleeping
   *      This means the queue does not contain any messages.  Any enqueue to
   *      the message queue will be informed that the queue was "sleeping". Only
   *      a single enqueuer will be informed that it woke the queue up, so this
   *      can be used to represent taking ownership of the queue.  Sleeping is
   *      represented with `back` containing `nullptr`.
   *
   *      Note that the queue can be empty and not asleep.
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
      STUB = 0x1,
      NOTIFY = 0x2,
      STATES = 0x3,
    };

    static constexpr uintptr_t MASK = ~static_cast<uintptr_t>(STATES);

    std::atomic<std::atomic<T*>*> back{nullptr};
    // The front is atomic as it can be read and written concurrently.
    std::atomic<T*> front{nullptr};

    T* get_containing_type(std::atomic<T*>* ptr)
    {
      static constexpr ptrdiff_t offset = offsetof(T, next);
      return snmalloc::pointer_offset_signed<T>(ptr, -offset);
    }

    template <typename TT>
    inline static bool has_state(TT* p, STATE f)
    {
      return ((uintptr_t)p & STATES) == f;
    }

    template <typename TT>
    inline static TT* set_state(TT* p, STATE f)
    {
      assert(is_clear(p));
      return (TT*)((uintptr_t)p | f);
    }

    template <typename TT>
    inline static bool is_clear(TT* p)
    {
      return clear_state(p) == p;
    }

    template <typename TT>
    inline static STATE get_state(TT* p)
    {
      return static_cast<STATE>((uintptr_t)p & STATES);
    }

    template <typename TT>
    static TT* clear_state(TT* p)
    {
      return (TT*)((uintptr_t)p & MASK);
    }

  public:
    constexpr MPSCQ() = default;

    T* peek_back()
    {
      return get_containing_type(clear_state(back.load(std::memory_order_relaxed)));
    }

    inline bool is_sleeping()
    {
      std::atomic<T*>* bk = back.load(std::memory_order_relaxed);

      return bk == nullptr;
    }

    /**
     * Enqueues (inserts) a message into the queue.
     *
     * Returns true if the queue was sleeping when the message was added.
     **/
    bool enqueue(T* t)
    {
      assert(is_clear(t));

      t->next.store(nullptr, std::memory_order_relaxed);
      auto* prev = back.exchange(&(t->next), std::memory_order_acq_rel);

      yield();

      // Pass on the notify info if set
      if (has_state(prev, NOTIFY))
      {
        t = set_state(t, NOTIFY);
        prev = clear_state(prev);
      }

      if (prev == nullptr)
      {
        // Was sleeping so use front to start list.
        // This releases permission to write to front to the consumer.
        front.store(t, std::memory_order_release);
        return true;
      }

      // This may be writing to `front` and in that case is this releases
      // permission to write to the consumer.
      prev->store(t, std::memory_order_release);
      return false;
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
    T* dequeue(snmalloc::Alloc& alloc, bool& notify)
    {
      // Returns the next message. If the next message
      // is not null, the front message is freed.

      T* fnt = front.load(std::memory_order_acquire);

      if (has_state(fnt, STUB))
      {
        fnt = clear_state(fnt);
        T* next = fnt->next.load(std::memory_order_relaxed);

        if (next == nullptr)
        {
          return nullptr;
        }

        notify = has_state(next, NOTIFY);
        next = clear_state(next);

        front = set_state(next, STUB);

        std::atomic_thread_fence(std::memory_order_acquire);

        alloc.dealloc(fnt, fnt->size());

        return next;
      }

      if (fnt == nullptr)
      {
        return nullptr;
      }

      notify = has_state(fnt, NOTIFY);
      fnt = clear_state(fnt);
      assert(fnt != nullptr);

      front.store(set_state(fnt, STUB), std::memory_order_relaxed);
      return fnt;
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
    T* dequeue(snmalloc::Alloc& alloc)
    {
      bool notify;
      return dequeue(alloc, notify);
    }

    /**
     * Used to find the first element in the queue. Only safe to use in the
     * consumer.
     **/
    T* peek()
    {
      auto fnt = front;
      if (has_state(fnt, STUB))
      {
        fnt = clear_state(fnt);
        return clear_state(fnt->next.load(std::memory_order_relaxed));
      }
      
      return clear_state(front);
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
     *   mark_notify; enqueue; mark_sleeping; dequeue
     * 
     * the mark sleeping will fail, and will not set its notify parameter, but the subsequent
     * dequeue will have its notify parameter set.
     *
     * State transition:
     *   NONE     -> NOTIFY;  return false
     *   SLEEPING -> NOTIFY;  return true
     *   NOTIFY   -> NOTIFY;  return false
     * Scheduling is required when the queue was SLEEPING, but not other states.
     */
    bool mark_notify()
    {
      std::atomic<T*>* bk = back.load(std::memory_order_relaxed);
      auto was_sleeping = false;

      while (true)
      {
        if (has_state(bk, NOTIFY))
        {
          break;
        }

        was_sleeping = bk == nullptr;
        std::atomic<T*>* next_val = was_sleeping ? &front : bk;

        next_val = set_state(next_val, NOTIFY);

        if (back.compare_exchange_strong(
              bk, next_val, std::memory_order_release))
        {
          break;
        }
      }

      return was_sleeping;
    }

    /**
     * If the queue is empty and has not been NOTIFIED, then this will
     * put the queue into the SLEEPING state.
     *
     * If the queue is non-empty, then this will return false.
     *
     * If the queue has been notified since the last enqueue, then 
     * it will return false, and will set the `notify` parameter to true.
     *
     * In the case that it actually puts the queue into the SLEEPING state,
     * any stub message if it exists will deleted.
     *
     * There are two types of empty queue that this code has to handle:
     *   1. back == &front   and  no STUB message
     *   2. front == back    and  STUB message
     * Only safe to call from the consumer.
     */
    bool mark_sleeping(snmalloc::Alloc& alloc, bool& notify)
    {
      UNUSED(alloc);

      std::atomic<T*>* bk = back.load(std::memory_order_relaxed);
      // Can't be sleeping already.
      assert(bk != nullptr);

      if (bk == &front)
      {
        return back.compare_exchange_strong(bk, nullptr, std::memory_order_release);
      }

      // Handle special cases for notify.
      if (get_state(bk) == NOTIFY)
      {
        // We have observed the queue to only contain a notify bit,
        // attempt to remove, so the notification can be handled.
        // Do not move to sleeping as we still need ownership to
        // handle the notification.  Also, we might be removing
        // the notification from a non-empty queue.
        notify =
          back.compare_exchange_strong(bk, clear_state(bk), std::memory_order_release);
        return false;
      }

      // Check for single stub entry
      T* fnt = front.load(std::memory_order_relaxed);
      if (set_state(get_containing_type(bk), STUB) == fnt)
      {
        front = nullptr;
        auto success = back.compare_exchange_strong(bk, nullptr, std::memory_order_release);
        if (success)
        {
          alloc.dealloc(clear_state(fnt));
          return true;
        }
        else
        {
          // Reset front as we failed to sleep the queue.
          front = fnt;
          return false;
        }
      }

      return false;
    }

    /**
     * Takes the queue out of the sleeping state.  Returns true, if it
     * succeeded in waking up the queue.
     */
    bool wake()
    {
      std::atomic<T*>* bk = back.load(std::memory_order_relaxed);

      if (bk != nullptr)
        return false;

      return back.compare_exchange_strong(
        bk, &front, std::memory_order_release);
    }
  };
} // namespace verona::rt
