// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once
#include "test/logging.h"

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
  template<typename T>
  class ThreadSyncSystematic
  {
    /// Model underlying locking provided by the handle.
    bool m = false;

    /// unpause incarnation
    /// Complete wrap around will lead to lost wake-up.  This seems safe to
    /// ignore.
    size_t unpause_incarnation = 0;

    void acquire()
    {
      auto guard = [&]() { return !m; };
      Systematic::yield_until(guard);
      m = true;
    }

  public:
    class ThreadSyncHandle
    {
      ThreadSyncSystematic& sync;
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

        auto incarnation = sync.unpause_incarnation;
        // Copy for capture by value
        auto sync_ptr = &sync;
        auto guard = [incarnation, sync_ptr]() {
          return incarnation != sync_ptr->unpause_incarnation;
        };
        // Guard should not hold here.
        assert(!guard());
        Systematic::yield_until(guard);

        sync.acquire();
      }

      ThreadSyncHandle(ThreadSyncSystematic& sync) : sync(sync) {}

      ~ThreadSyncHandle()
      {
        assert(sync.m == true);
        sync.m = false;

        if (wake_on_exit)
        {
          // Treat as a yield pointer if thread is under systematic testing
          // control.
          sync.unpause_incarnation++;
          Systematic::yield();
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
      UNUSED(me);
      acquire();
      return ThreadSyncHandle(*this);
    }

    /**
     * This unpauses all threads.
     */
    void unpause_all(T* me)
    {
      handle(me).unpause_all();
    }
  };
}
