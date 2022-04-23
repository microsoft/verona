// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include <cassert>
#include <chrono>
#include <list>
#include <test/opt.h>
#include <verona.h>

using namespace verona::rt;
using namespace verona::rt::api;
using namespace std::chrono;

extern "C" inline void dump_flight_recorder()
{
  Logging::SysLog::dump_flight_recorder();
}

#define check(x) \
  if (!(x)) \
  { \
    printf("Failed %s:%d - check(%s)\n", __FILE__, __LINE__, #x); \
    fflush(stdout); \
    abort(); \
  }

/**
 * Implements a busy loop that spins for the specified number of microseconds.
 *
 * This is used instead of sleep as it keeps the core busy for the specified
 * time, and hence can be used in a test to emulate work.
 *
 * Sleep cannot be used to emulate work as many threads can be sleeping at once
 * and thus it appears to be quicker.  E.g. 2000 threads sleeping for 1 ms, can
 * occur in 1 ms on a single core box, where as 2000 threads calling
 * busy_loop(1'000) would have to take 2 seconds to complete.
 */
inline void busy_loop(size_t u_sec)
{
  auto wait = [](size_t step_u_sec) {
    std::chrono::microseconds usec(step_u_sec);
    auto start = std::chrono::steady_clock::now();
    auto end = start + usec;

    // spin
    while (std::chrono::steady_clock::now() <= end)
      ;
  };

  size_t it_count = u_sec / 10;
  // Break into multiple shorter waits so that pre-emption can be detected.
  // This is not perfect, but it is good enough for benchmarking.
  for (size_t j = 0; j < it_count; j++)
  {
    wait(10);
  }

  wait(u_sec % 10);
}

class SystematicTestHarness
{
  size_t seed = 0;
  /**
   * External threads created during execution can only be joined once
   * sched.run() is finished. Not joining on these threads can lead to a race
   * between their destruction and operations such as
   * snmalloc::debug_check_empty. Since the test has no way to detect when the
   * execution has finished, we make the harness responsible for tracking and
   * joining on external threads.
   *
   * Storage type is verona::PlatformThread, so that the harness can be reused
   * on platforms with custom threading.
   */
  std::list<PlatformThread> external_threads;

public:
  opt::Opt opt;

  bool detect_leaks;
  size_t cores;
  size_t seed_lower;
  size_t seed_upper;
  high_resolution_clock::time_point start;

  SystematicTestHarness(int argc, const char* const* argv) : opt(argc, argv)
  {
    std::cout << "Harness starting." << std::endl;

    for (int i = 0; i < argc; i++)
    {
      std::cout << " " << argv[i];
    }

#ifdef USE_SYSTEMATIC_TESTING
    size_t count = opt.is<size_t>("--seed_count", 100);
#else
    size_t count = opt.is<size_t>("--seed_count", 1);
#endif

    // Detect if seed supplied.  If not, then generate a seed, and add to
    // command line print out.
    if (opt.has("--seed"))
    {
      seed_lower = opt.is<size_t>("--seed", 0);
    }
    else
    {
      seed_lower = ((snmalloc::Aal::tick()) & 0xffffffff) * count;
      std::cout << " --seed " << seed_lower;
    }

    std::cout << std::endl;

    start = high_resolution_clock::now();
    seed_upper = seed_lower + count;

#if defined(USE_FLIGHT_RECORDER) || defined(CI_BUILD)
    Logging::enable_crash_logging();
#endif

    if (opt.has("--log-all") || (seed_lower + 1 == seed_upper))
      Logging::enable_logging();

    cores = opt.is<size_t>("--cores", 4);

    detect_leaks = !opt.has("--allow_leaks");
    Scheduler::set_detect_leaks(detect_leaks);

#if defined(_WIN32) && defined(CI_BUILD)
    _set_error_mode(_OUT_TO_STDERR);
    _set_abort_behavior(0, _WRITE_ABORT_MSG);
#endif
  }

  template<typename... Args>
  void run(void f(Args...), Args... args)
  {
    for (seed = seed_lower; seed < seed_upper; seed++)
    {
      std::cout << "Seed: " << seed << std::endl;

      Scheduler& sched = Scheduler::get();
#ifdef USE_SYSTEMATIC_TESTING
      Systematic::set_seed(seed);
      if (seed % 2 == 1)
      {
        sched.set_fair(true);
      }
      else
      {
        sched.set_fair(false);
      }
#else
      UNUSED(seed);
#endif
      sched.init(cores);

      f(std::forward<Args>(args)...);

      sched.run();

      // Join on all created external threads and clear the list.
      while (!external_threads.empty())
      {
        auto& thread = external_threads.front();
        thread.join();
        external_threads.pop_front();
      }

      if (detect_leaks)
        snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
      high_resolution_clock::time_point t1 = high_resolution_clock::now();
      std::cout << "Time so far: "
                << duration_cast<seconds>((t1 - start)).count() << " seconds"
                << std::endl;
    }

    std::cout << "Test Harness Finished!" << std::endl;
  }

  /**
   * Add an external thread to the system, which will be joined after
   * sched.run() finishes. Do not create any verona::PlatformThread or
   * std::thread explicitly in a test when using SystematicTestHarness.
   *
   * Same arguments as the std::thread constructor.
   */
  template<typename F, typename... Args>
  void external_thread(F&& f, Args&&... args)
  {
    // TODO Thread ID
    // Pre-inject the thread into systematic testing.  This must be done
    // before the thread is created, so that it location in systematic
    // testing is deterministic.
    Systematic::Local* t = Systematic::create_systematic_thread(0);

    auto f_wrap = [t](F&& f, Args&&... args) {
      // Before running any code join systematic testing
      Systematic::attach_systematic_thread(t);
      f(args...);
      // Leave systematic testing.
      Systematic::finished_thread();
    };

    external_threads.emplace_back(f_wrap, f, args...);
  }

  size_t current_seed()
  {
    check(seed != 0);
    return seed;
  }
};
