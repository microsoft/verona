// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once
#include "test/systematic.h"

#include <condition_variable>
#include <mutex>

/**
 * This file contains the synchronisation implementation for suspending
 * and resuming threads.  It is design to be a thin wrapper that could
 * be replaced with alternatives for other platforms.
 *
 * This implementation provides systematic testing.
 */
namespace verona::rt
{
  enum class SystematicState
  {
    Active,
    Finished
  };

  inline snmalloc::function_ref<bool()> true_thunk{[]() { return true; }};

  struct LocalSync
  {
    /// Used by systematic testing to implement the condition variable,
    /// and thread termination.
    SystematicState systematic_state = SystematicState::Active;

    /// Used to specify a condition when this thread should/could make
    /// progress.  It is used to implement condition variables.
    snmalloc::function_ref<bool()> guard = true_thunk;

    /// How many uninterrupted steps this threads has been selected to run for.
    size_t steps = 0;

    /// Alters distribution of steps taken in systematic testing.
    size_t systematic_speed_mask = 1;

    /// Used to hold thread asleep.
    std::condition_variable cv;
  };

  template<typename T>
  class ThreadSyncSystematic
  {
    /// Mutex for manipulating systematic testing datastructures
    ///  * running_thread
    ///  * systematic_status
    ///  * shutdown
    std::mutex m_sys;

    /// Model underlying locking provided by the handle.
    std::atomic<bool> m = false;

    /// unpause incarnation
    /// Complete wrap around will lead to lost wake-up.  This seems safe to
    /// ignore.
    std::atomic<size_t> unpause_incarnation = 0;

    /// Specifies which thread is currently executing in systematic testing
    /// nullptr is used to mean no thread is currently running.
    T* running_thread = nullptr;

    /// Specifies the first thread to consider when choosing a thread, if
    /// being launch from outside.
    T* first_thread = nullptr;

    /// Used to prevent systematic testing attempt to access threads when the
    /// runtime has been deallocated.
    bool shutdown = true;

    /// Must hold the systematic testing lock to call this.
    /// External is used for calls from outside a scheduler thread.
    void
    choose_thread(std::unique_lock<std::mutex>&, T* me, bool external = false)
    {
      if (external && running_thread != nullptr)
      {
        // Runtime was not asleep, so don't try to wake up a thread.
        return;
      }

      auto r = Systematic::get_prng_next();
      auto i = snmalloc::bits::ctz(r != 0 ? r : 1);
      auto start = running_thread;
      assert(me == start);
      UNUSED(me);
      if (start == nullptr)
      {
        // This is an external wake up
        start = first_thread;
      }

      // Skip to a first choice for selecting.
      for (; i > 0; i--)
        start = start->next;

      auto result = start;
      while ((result->local_sync.systematic_state != SystematicState::Active) ||
             !result->local_sync.guard())
      {
        result = result->next;
        if (result == start)
        {
          // The following note is for anyone wanting to add an assertion about
          // the conditions that could be true at this point.
          //
          // Note, this could have zero external event sources, with the runtime
          // waking up from pausing.  The external thread that dropped the event
          // source, can still be `unpausing` the runtime.  So it appears all
          // threads are going to sleep, but the external thread will continue
          // to wake them until `runtime_pausing` is unset.

          running_thread = nullptr;
          Systematic::cout() << "All threads sleeping!" << Systematic::endl;
          return;
        }
      }
      Systematic::cout() << "Set running thread:" << result->systematic_id
                         << Systematic::endl;
      assert(result->local_sync.guard());

      running_thread = result;
      assert(result->local_sync.systematic_state == SystematicState::Active);
      result->local_sync.cv.notify_all();
    }

    void wait_for_my_turn_inner(std::unique_lock<std::mutex>& lock, T* me)
    {
      assert(lock.mutex() == &m_sys);
      Systematic::cout() << "Waiting for turn" << Systematic::endl;
      while (running_thread != me)
        me->local_sync.cv.wait(lock);
      assert(me->local_sync.systematic_state == SystematicState::Active);
    }

    /// Must hold the systematic testing lock to call this.
    /// Will only pass control back to this thread once the guard g has been
    /// established.
    void yield_until(
      T*& me,
      std::unique_lock<std::mutex>& lock,
      snmalloc::function_ref<bool()> g)
    {
      assert(lock.mutex() == &m_sys);
      me->local_sync.guard = g;
      choose_thread(lock, me);
      wait_for_my_turn_inner(lock, me);
      me->local_sync.guard = true_thunk;
    }

    void acquire(T* me)
    {
      auto guard = [&]() { return !m; };
      while (true)
      {
        std::unique_lock<std::mutex> lock(m_sys);
        // External threads will just have to spin here.
        if (m)
        {
          if (me != nullptr)
          {
            yield_until(me, lock, guard);
          }
          continue;
        }
        m = true;
        break;
      }
    }

  public:
    class ThreadSyncHandle
    {
      ThreadSyncSystematic& sync;
      T* me;
      bool wake_on_exit = false;

    public:
      /**
       * Wake up all threads in the thread pool
       *
       * Will only occur once the handle is dropped.
       */
      void unpause_all()
      {
        wake_on_exit = true;
      }

      /**
       * Pause this thread
       */
      void pause()
      {
        assert(sync.m == true);
        sync.m = false;
        assert(me->local_sync.systematic_state == SystematicState::Active);
        {
          std::unique_lock<std::mutex> lock(sync.m_sys);
          auto incarnation = sync.unpause_incarnation.load();
          // Copy for capture by value
          auto sync_ptr = &sync;
          auto guard = [incarnation, sync_ptr]() {
            return incarnation != sync_ptr->unpause_incarnation.load();
          };
          // Guard should not hold here.
          assert(!guard());
          sync.yield_until(me, lock, guard);
        }

        assert(sync.running_thread == me);
        sync.acquire(me);
      }

      ThreadSyncHandle(ThreadSyncSystematic& sync, T* me) : sync(sync), me(me)
      {}

      ~ThreadSyncHandle()
      {
        sync.m = false;

        if (me == nullptr)
        {
          // All the threads could be sleeping waiting for this lock.
          std::unique_lock<std::mutex> lock(sync.m_sys);
          if (sync.shutdown)
            return;
          sync.choose_thread(lock, me, true);
        }

        if (wake_on_exit)
        {
          // Treat as a yield pointer if thread is under systematic testing
          // control.
          if (me != nullptr)
          {
            Systematic::cout() << "unpause internal" << Systematic::endl;
            sync.unpause_incarnation++;
            sync.yield(me);
          }
          else
          {
            Systematic::cout() << "unpause external" << Systematic::endl;
            // Can be signalled from outside the runtime if external work is
            // injected. If this is a runtime thread, then yield.
            // This will wake a thread if none are currently running, otherwise
            // does nothing.
            // m_sys mutex is required to prevent lost wake-up
            std::unique_lock<std::mutex> lock(sync.m_sys);
            if (sync.shutdown)
              return;
            sync.unpause_incarnation++;
            sync.choose_thread(lock, me, true);
          }
        }
      }
    };

    /**
     * Call this to begin modifying the ThreadSync
     *
     * The ThreadSyncHandle provides single threaded access to pausing and
     * waking threads, and thus can be used as a lock.
     */
    ThreadSyncHandle handle(T* me)
    {
      acquire(me);
      return ThreadSyncHandle(*this, me);
    }

    /**
     * This unpauses all threads.
     */
    void unpause_all(T* me)
    {
      handle(me).unpause_all();
    }

    /**
     * Call this when the thread has completed.
     */
    void thread_finished(T* me)
    {
      std::unique_lock<std::mutex> lock(m_sys);
      assert(me->local_sync.systematic_state == SystematicState::Active);
      me->local_sync.systematic_state = SystematicState::Finished;

      assert(running_thread == me);

      Systematic::cout() << "Thread finished." << Systematic::endl;

      // Confirm at least one other thread is running,
      auto curr = me;
      while (curr->local_sync.systematic_state != SystematicState::Active)
      {
        curr = curr->next;
        if (curr == me)
        {
          Systematic::cout() << "Last thread finished." << Systematic::endl;
          // No threads left
          running_thread = nullptr;
          shutdown = true;
          return;
        }
      }
      choose_thread(lock, me);
    };

    /**
     * Call this when the thread starts.
     */
    void thread_start(T* me)
    {
      assert(me != nullptr);
      std::unique_lock<std::mutex> lock(m_sys);
      wait_for_my_turn_inner(lock, me);
    };

    /**
     * Call this to potentially yield control
     *
     * Only used by systematic testing.
     */
    void yield(T* me)
    {
      if (me == nullptr)
        return;

      assert(running_thread == me);

      if (me->local_sync.steps == 0)
      {
        std::unique_lock<std::mutex> lock(m_sys);
        yield_until(me, lock, true_thunk);
        me->local_sync.steps =
          Systematic::get_prng_next() & me->local_sync.systematic_speed_mask;
      }
      else
      {
        me->local_sync.steps--;
      }
    };

    void init(T* first)
    {
      std::unique_lock<std::mutex> lock(m_sys);
      running_thread = nullptr;
      first_thread = first;
      shutdown = false;
      choose_thread(lock, nullptr, true);
    }

    void reset()
    {
      std::unique_lock<std::mutex> lock(m_sys);
      first_thread = nullptr;
    }
  };
}
