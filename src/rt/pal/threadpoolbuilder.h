// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "threading.h"

#include <list>

/**
 * This constructs a platforms affinitised set of threads.
 */
namespace verona::rt
{
  class ThreadPoolBuilder
  {
    std::list<PlatformThread> threads;
    size_t thread_count;
    size_t index = 0;

    template<typename... Args>
    void add_thread_impl(void (*body)(Args...), Args... args)
    {
      if (index != thread_count)
      {
        threads.emplace_back(body, args...);
      }
      else
      {
        Systematic::start();
        body(args...);
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
    ThreadPoolBuilder(size_t thread_count)
    {
      this->thread_count = thread_count - 1;
    }

    /**
     * Add a thread to run in this thread pool.
     */
    template<typename... Args>
    void add_thread(size_t affinity, void (*body)(Args...), Args... args)
    {
#ifdef USE_SYSTEMATIC_TESTING
      // Don't use affinity with systematic testing.  We're only ever running
      // one thread at a time in systematic testing mode and by pinning each
      // thread to a core we massively increase contention.
      UNUSED(affinity);
      add_thread_impl(body, args...);
#else
      add_thread_impl(&run_with_affinity, affinity, body, args...);
#endif
      index++;
    }

    /**
     * The destructor waits for all threads to finish, and
     * then tidies up.
     *
     *  The number of executions is one larger than the number of threads
     * created as there is also the main thread.
     */
    ~ThreadPoolBuilder()
    {
      assert(index == thread_count + 1);

      while (!threads.empty())
      {
        auto& thread = threads.front();
        thread.join();
        threads.pop_front();
      }
    }
  };
}
