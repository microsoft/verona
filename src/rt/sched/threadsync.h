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
 * This is the standard implementation based on the C++ standard
 * libraries condition variables.
 */
namespace verona::rt
{
  template<class T>
  class ThreadSync
  {
    std::mutex m;
    std::condition_variable cv;

  public:
    class ThreadSyncHandle
    {
      ThreadSync& sync;
      std::unique_lock<std::mutex> lock;
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
        sync.cv.wait(lock);
      }

      ThreadSyncHandle(ThreadSync& sync) : sync(sync), lock(sync.m) {}

      ~ThreadSyncHandle()
      {
        if (wake_on_exit)
        {
          lock.unlock();
          sync.cv.notify_all();
        }
      }
    };

    /**
     * Call this to begin modifying the ThreadSync
     *
     * The ThreadSyncHandle provides single threaded access to pausing and
     * waking threads, and thus can be used as a lock.
     */
    ThreadSyncHandle handle(T*)
    {
      return ThreadSyncHandle(*this);
    }

    /**
     * Call this when the thread has completed.
     */
    inline void thread_finished(T*){};

    /**
     * Call this when the thread starts.
     */
    inline void thread_start(T*){};

    /**
     * Call this to potentially yield control
     *
     * Only used by systematic testing.
     */
    inline void yield(T*){};

    void init(T*) {}

    void reset() {}
  };
}