// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once
#include "../pal/semaphore.h"
#include "test/logging.h"

/**
 * This file contains the synchronisation implementation for suspending
 * and resuming threads.  It is design to be a thin wrapper that could
 * be replaced with alternatives for other platforms.
 *
 * This is the standard implementation based on the C++ standard
 * libraries semaphores.
 */
namespace verona::rt
{
  /**
   * This class is a custom spin lock for handling thread pausing and unpausing.
   *
   * It uses three states:
   *  - Unlocked: No thread is currently holding the lock.
   *  - Locked: A thread is holding the lock, and is processing its
   *    pause/unpause request
   *  - LockedUnpauseNeeded: A thread is holding the lock, and another thread
   *    has attempted to lock for unpause. Rather than waiting for the lock to
   *    become available, the thread currently holding the lock takes
   *    responsibility for unpausing the other threads.
   */
  class SchedulerLock
  {
    enum State
    {
      Unlocked,
      Locked,
      LockedUnpauseNeeded
    };

    std::atomic<State> state{Unlocked};

  public:
    /**
     * Acquires the lock. Spins waiting for it to be available.
     */
    void lock()
    {
      Logging::cout() << "Locking Scheduler." << Logging::endl;
      auto u = Unlocked;
      while (!state.compare_exchange_strong(u, Locked))
      {
        // Have to reset u, as compare exchange gives it the current value.
        u = Unlocked;
        while (state.load(std::memory_order_acquire) != Unlocked)
        {
          snmalloc::Aal::pause();
        }
      }
      Logging::cout() << "Locking Scheduler done" << Logging::endl;
    }

    /**
     * Attempts to release the lock.  If the lock has received an unpause
     * request, then will return false, and continues to hold the lock.
     * Otherwise, will return true, and the lock is actually released.
     */
    bool unlock()
    {
      assert(state.load(std::memory_order_relaxed) != Unlocked);
      auto l = Locked;
      return state.compare_exchange_strong(l, Unlocked);
    }

    /**
     * Releases the lock ignoring any unpause requests.
     */
    void unlock_unpause()
    {
      assert(state.load(std::memory_order_relaxed) == LockedUnpauseNeeded);
      state.store(Unlocked);
    }

    /**
     * Attempt to acquire the lock to unpause threads.
     * Returns true, if it acquires the lock, and false if it
     * notified the thread currently holding the thread to wake up waiters.
     */
    bool lock_for_unpause()
    {
      return state.exchange(LockedUnpauseNeeded) == Unlocked;
    }
  };

  struct LocalSync
  {
    pal::SleepHandle sem;
    LocalSync* next{nullptr};
  };

  template<class T>
  class ThreadSync
  {
    SchedulerLock lock;
    LocalSync* waiters = nullptr;

    void unlock()
    {
      Logging::cout() << "Unlock Scheduler lock" << Logging::endl;

      // Releasing the lock can pickup an unpause request
      if (!lock.unlock())
      {
        Logging::cout() << "Pending unpause" << Logging::endl;

        auto* curr = waiters;
        waiters = nullptr;
        // Subsequent unpause requests can be ignored as there are no more
        // waiters as we have held the lock continuously.
        lock.unlock_unpause();
        // Don't need to hold the lock to wake up the waiters.
        while (curr != nullptr)
        {
          auto next = curr->next;
          curr->sem.wake();
          curr = next;
        }
      }
    }

  public:
    void unpause_all(T*)
    {
      Logging::cout() << "Unpause all" << Logging::endl;
      if (lock.lock_for_unpause())
        unlock();
      Logging::cout() << "Unpause all done" << Logging::endl;
    }

    class ThreadSyncHandle
    {
      T* thread;
      ThreadSync& sync;
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
       *
       * If this is the last thread, then something external must call
       * unpause_all to restart the paused threads.
       */
      void pause()
      {
        Logging::cout() << "Add to list of waiters" << Logging::endl;
        thread->local_sync.next = sync.waiters;
        sync.waiters = &(thread->local_sync);
        sync.unlock();

        Logging::cout() << "Sleep" << Logging::endl;
        thread->local_sync.sem.sleep();
        Logging::cout() << "Awake!" << Logging::endl;

        sync.lock.lock();
      }

      ThreadSyncHandle(T* thread, ThreadSync& sync) : thread(thread), sync(sync)
      {
        sync.lock.lock();
      }

      ~ThreadSyncHandle()
      {
        if (wake_on_exit)
        {
          // Set to the unpause state. We already hold the lock, so we
          // know that it will return false.
          sync.lock.lock_for_unpause();
        }
        sync.unlock();
      }
    };

    /**
     * Call this to begin modifying the ThreadSync
     *
     * The ThreadSyncHandle provides single threaded access to pausing and
     * waking threads, and thus can be used as a lock.
     */
    ThreadSyncHandle handle(T* t)
    {
      return ThreadSyncHandle(t, *this);
    }
  };
}
