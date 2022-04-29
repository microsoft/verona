#pragma once

#include <atomic>

#include "mpmcq.h"
#include "schedulerstats.h"

namespace verona::rt
{
  template<class T>
    class Core
    {
      public:
        size_t affinity = 0;
        T* token_cown = nullptr;
        MPMCQ<T> q;
        std::atomic<Core<T>*> next = nullptr;

        /// Progress and synchronization between the threads.
        //  These counters represent progress on a CPU core, not necessarily on
        //  the core's queue. This is necessary to take into account core-stealing
        //  to avoid spawning many threads on a core hogged by a long running
        //  behaviour but with an empty cown queue.
        std::atomic<std::size_t> progress_counter = 0;
        std::atomic<std::size_t> servicing_threads = 0;
        std::atomic<std::size_t> last_worker = 0;

        
        /// Moved from the scheduler thread
        std::atomic<size_t> total_cowns = 0;
        std::atomic<size_t> free_cowns = 0;
        SchedulerStats stats;


      public:
        Core() : token_cown{T::create_token_cown()}, q{token_cown}
        {
          // Let the thread set the owning core.
        }

        ~Core() {}
    };
}
