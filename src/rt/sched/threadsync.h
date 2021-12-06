// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once
#include "test/systematic.h"

#include "../pal/semaphore.h"
#include <mutex>

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
  struct LocalSync
  {
    pal::Semaphore sem;
    LocalSync* next{nullptr};
  };
  
  template<class T>
  class ThreadSync
  {
    std::mutex m;
    LocalSync* waiters = nullptr;

    void unpause_all()
    {
      auto* curr = waiters;
      while (curr != nullptr)
      {
        curr->sem.release();
        curr = curr->next;
      }
      waiters = nullptr;
    }

  public:
    class ThreadSyncHandle
    {      
      T* thread;
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
        Systematic::cout() << "Increasing sleep count" << Systematic::endl;
        thread->local_sync.next = sync.waiters;
        sync.waiters = &(thread->local_sync);
        lock.unlock();

        Systematic::cout() << "Acquire Sem" << Systematic::endl;
        thread->local_sync.sem.acquire();
        Systematic::cout() << "Acquired Sem!" << Systematic::endl;

        lock.lock();
      }

      ThreadSyncHandle(T* thread, ThreadSync& sync) : thread(thread), sync(sync), lock(sync.m) {}

      ~ThreadSyncHandle()
      {
        if (wake_on_exit)
        {
          sync.unpause_all();
          lock.unlock();
        }
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