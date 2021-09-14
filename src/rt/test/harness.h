// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include <cassert>
#include <chrono>
#include <list>
#include <test/opt.h>
#include <verona.h>

using namespace verona::rt;
using namespace std::chrono;

extern "C" void dump_flight_recorder()
{
  Systematic::SysLog::dump_flight_recorder();
}

#define check(x) \
  if (!(x)) \
  { \
    printf("Failed %s:%d - check(%s)\n", __FILE__, __LINE__, #x); \
    fflush(stdout); \
    abort(); \
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

    size_t count = opt.is<size_t>("--seed_count", 1);

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
    Systematic::enable_crash_logging();
#endif

    if (opt.has("--log-all") || (seed_lower + 1 == seed_upper))
      Systematic::enable_logging();

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
    external_threads.emplace_back(f, args...);
  }

  size_t current_seed()
  {
    check(seed != 0);
    return seed;
  }
};
