// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <chrono>
#include <string>
#include <atomic>

namespace verona::ffi
{
  /// Reports time elapsed between creation and destruction.
  ///
  /// Use:
  ///  {
  ///    auto T = TimeReport("My action");
  ///    ... some action ...
  ///  }
  ///  // Here prints "My action: 12ms" to stderr
  class TimeReport
  {
    timespec start;
    std::string name;
#ifdef CLOCK_PROF
    static const clockid_t clock = CLOCK_PROF;
#else
    const clockid_t clock = CLOCK_PROCESS_CPUTIME_ID;
#endif

  public:
    TimeReport(std::string n) : name(n)
    {
      std::atomic_signal_fence(std::memory_order::memory_order_seq_cst);
      clock_gettime(clock, &start);
    }
    ~TimeReport()
    {
      using namespace std::chrono;
      timespec end;
      std::atomic_signal_fence(std::memory_order::memory_order_seq_cst);
      clock_gettime(clock, &end);
      std::atomic_signal_fence(std::memory_order::memory_order_seq_cst);
      auto interval_from_timespec = [](timespec t) {
        return seconds{t.tv_sec} + nanoseconds{t.tv_nsec};
      };
      auto elapsed =
        interval_from_timespec(end) - interval_from_timespec(start);

      // TODO: Avoid printfs on headers
      fprintf(
        stderr,
        "%s: %ldms\n",
        name.c_str(),
        duration_cast<milliseconds>(elapsed).count());
    }
  };
} // namespace verona::ffi
