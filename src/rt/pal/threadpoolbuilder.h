// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "cpu.h"

#include <thread>

/**
 * This constructs a platforms affinitised set of threads.
 */
namespace verona::rt
{
  class ThreadPoolBuilder
  {
    Topology topology;
    std::thread* threads;
    size_t thread_count;
    size_t index = 0;

    template<typename... Args>
    void add_thread_impl(void (*body)(Args...), Args... args)
    {
      if (index < thread_count)
      {
        threads[index] = std::thread(body, args...);
      }
      else
      {
        abort();
      }
    }

    template<typename... Args>
    static void
    run_with_affinity(size_t affinity, void (*body)(Args...), Args... args)
    {
      cpu::set_affinity(affinity);
      body(args...);
    }

  public:
    ThreadPoolBuilder(size_t thread_count) : thread_count(thread_count)
    {
      threads = new std::thread[thread_count];
      topology.acquire();
    }

    /**
     * Add a thread to run in this thread pool.
     */
    template<typename... Args>
    void add_thread(void (*body)(Args...), Args... args)
    {
#ifdef USE_SYSTEMATIC_TESTING
      // Don't use affinity with systematic testing.  We're only ever running
      // one thread at a time in systematic testing mode and by pinning each
      // thread to a core we massively increase contention.
      add_thread_impl(body, args...);
#else
      add_thread_impl(&run_with_affinity, topology.get(index), body, args...);
#endif
      index++;
    }

    /**
     * The destructor waits for all threads to finish, and
     * then tidies up.
     */
    ~ThreadPoolBuilder()
    {
      assert(index == thread_count);
      for (size_t i = 0; i < thread_count; i++)
      {
        threads[i].join();
      }

      for (size_t i = 0; i < thread_count; i++)
      {
        threads[i].~thread();
      }

      delete[] threads;

      topology.release();
    }
  };
}