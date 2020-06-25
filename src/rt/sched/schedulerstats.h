// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include <iostream>
#include <snmalloc.h>

namespace verona::rt
{
  using namespace snmalloc;
  class SchedulerStats
  {
  private:
#ifdef USE_SCHED_STATS
    size_t steal_count = 0;
    size_t pause_count = 0;
    std::atomic<size_t> unpause_count = 0;
    std::atomic<size_t> lifo_count = 0;
#endif

  public:
    ~SchedulerStats()
#ifdef USE_SCHED_STATS
    {
      static std::atomic_flag lock = ATOMIC_FLAG_INIT;
      static SchedulerStats global;

      if (this != &global)
      {
        FlagLock f(lock);
        global.add(*this);
      }
      else
      {
        print(std::cout);
      }
    }
#else
      = default;
#endif

    void steal()
    {
#ifdef USE_SCHED_STATS
      steal_count++;
#endif
    }

    void pause()
    {
#ifdef USE_SCHED_STATS
      pause_count++;
#endif
    }

    void unpause()
    {
#ifdef USE_SCHED_STATS
      unpause_count++;
#endif
    }

    void lifo()
    {
#ifdef USE_SCHED_STATS
      lifo_count++;
#endif
    }

    void add(SchedulerStats& that)
    {
      UNUSED(that);

#ifdef USE_SCHED_STATS
      steal_count += that.steal_count;
      pause_count += that.pause_count;
      unpause_count += that.unpause_count;
      lifo_count += that.lifo_count;
#endif
    }

    void print(std::ostream& o, uint64_t dumpid = 0)
    {
      UNUSED(o);
      UNUSED(dumpid);

#ifdef USE_SCHED_STATS
      CSVStream csv(&o);

      if (dumpid == 0)
      {
        // Output headers for initial dump
        // Keep in sync with data dump
        csv << "SchedulerStats"
            << "DumpID"
            << "Steal"
            << "LIFO"
            << "Pause"
            << "Unpause" << csv.endl;
      }

      csv << "SchedulerStats" << dumpid << steal_count << lifo_count
          << pause_count << unpause_count << csv.endl;
#endif
    }
  };
} // namespace verona::rt
