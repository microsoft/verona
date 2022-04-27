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
    
    public:
      SysMonitor(SysMonitor const&) = delete;

      static SysMonitor<Scheduler>& get()
      {
        static SysMonitor<Scheduler> instance;
        return instance;
      }

      template<class Pool>
      static void run_monitor(Pool* pool, ThreadPoolBuilder& builder)
      {
        UNUSED(builder);
        using namespace std::chrono_literals;
        assert(pool != nullptr);
        assert(pool->core_count != 0);
        auto& scheduler = Scheduler::get(); 
        /// TODO expose this as a tunable parameter
        auto quantum = 10ms;
        while(!SysMonitor::get().done)
        {
          size_t scan[pool->core_count]; 
          for (size_t i = 0; i < pool->core_count; i++)
          {
            scan[i] = pool->cores[i]->progress_counter;
          }
          std::this_thread::sleep_for(quantum);
          if (SysMonitor::get().done)
          {
            return;
          }
          // Look for progress 
          for (size_t i = 0; i < pool->core_count; i++)
          {
            size_t count = pool->cores[i]->progress_counter; 
            if (scan[i] == count)
            {
              // We pass the count as argument in case there was some progress
              // in the meantime.
              Scheduler::get().spawnThread(pool->cores[i], count);
            }
          }
        }
      }


    
  };
}
