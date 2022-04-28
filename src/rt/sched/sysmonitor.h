#pragma once

#include "pal/threadpoolbuilder.h"

#include <atomic>
#include <chrono>
#include <cassert>

namespace verona::rt
{
  // Singleton class monitoring the progress on cores.
  // Whenever the SysMonitor detects a hogged core, it 
  // calls into the threadpool to schedule a new thread on that 
  // particular core.
  template<class Scheduler>
  class SysMonitor
  {
    private:
      friend Scheduler;
      /// When true, the SysMonitor should stop.
      std::atomic_bool done = false;

      SysMonitor() {}
    
    public:
      SysMonitor(SysMonitor const&) = delete;

      static SysMonitor<Scheduler>& get()
      {
        static SysMonitor<Scheduler> instance;
        return instance;
      }

      void run_monitor(ThreadPoolBuilder& builder)
      {
        using namespace std::chrono_literals;
        auto* pool = Scheduler::get().core_pool;
        assert(pool != nullptr);
        assert(pool->core_count != 0);

        // TODO expose this as a tunable parameter
        auto quantum = 10ms;
        while(!done)
        {
          size_t scan[pool->core_count]; 
          for (size_t i = 0; i < pool->core_count; i++)
          {
            scan[i] = pool->cores[i]->progress_counter;
          }
          std::this_thread::sleep_for(quantum);
          if (done)
          {
            return;
          }
          // Look for progress 
          for (size_t i = 0; i < pool->core_count; i++)
          {
            size_t count = pool->cores[i]->progress_counter; 
            // Counter is the same and there is some work to do.
            if (scan[i] == count && !pool->cores[i]->q.nothing_old())
            {
              // We pass the count as argument in case there was some progress
              // in the meantime.
              Scheduler::get().spawnThread(builder, pool->cores[i], count);
            }
          }
        }
        // Wake all the threads stuck on their condition vars.

        // As this thread is the only one modifying the builder thread list,
        // we know nothing should be modifying it right now and can thus exit 
        // to join on every single thread.
      }

      void threadExit()
      {
        // if a thread exits, we are done
        done = true;
      }
  };
}
