// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "mpmcq.h"
#include "schedulerstats.h"

#include <atomic>
#include <snmalloc/snmalloc.h>

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

    SchedulerStats stats;

  public:
    Core() : token_cown{T::create_token_cown()}, q{token_cown}
    {
      token_cown->set_token_owning_core(this);
    }

    ~Core() {}
  };
}
