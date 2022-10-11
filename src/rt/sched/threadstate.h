// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include <atomic>
#include <snmalloc/snmalloc.h>

namespace verona::rt
{
  class ThreadState
  {
  public:
    // ThreadState counters.
    struct StateCounters
    {
      size_t active_threads{0};
      std::atomic<size_t> barrier_count{0};

      constexpr StateCounters() = default;
    };

  private:
    StateCounters internal_state;

  public:
    constexpr ThreadState() = default;

    void set_barrier(size_t thread_count)
    {
      internal_state.barrier_count = thread_count;
      internal_state.active_threads = thread_count;
    }

    /// @warn Should be holding the threadpool lock.
    size_t exit_thread()
    {
      return internal_state.barrier_count.fetch_sub(1) - 1;
    }

    size_t get_active_threads()
    {
      return internal_state.active_threads;
    }

    void dec_active_threads()
    {
      internal_state.active_threads--;
    }

    void inc_active_threads()
    {
      internal_state.active_threads++;
    }
  };
} // namespace verona::rt
