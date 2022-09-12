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

    // The number of cowns in the per-core list `list`.
    std::atomic<size_t> total_cowns = 0;

    // The number of cowns that have been collected in the per-thread list
    // `list`. This is atomic as other threads can collect the body of the
    // cown managed from this thread.  They cannot collect the actual cown
    // allocation.  The ratio of free_cowns to total_cowns is used to
    // determine when to walk the `list` to collect the stubs.
    std::atomic<size_t> free_cowns = 0;

    SchedulerStats stats;

    std::atomic<T*> list = nullptr;

  public:
    Core() : token_cown{T::create_token_cown()}, q{token_cown}
    {
      token_cown->set_owning_core(this);
    }

    ~Core() {}

    void collect(Alloc& alloc)
    {
      T* head = list.exchange(nullptr);
      T* tail = head;
      T* cown = head;
      while (cown != nullptr)
      {
        if (!cown->is_collected())
          cown->collect(alloc);
        tail = cown;
        cown = cown->next;
      }
      if (tail != nullptr)
        add_cowns(head, tail);
    }

    void try_collect(Alloc& alloc, EpochMark epoch)
    {
      T* head = list.exchange(nullptr);
      T* tail = head;
      T* cown = head;
      while (cown != nullptr)
      {
        T* n = cown->next;
        cown->try_collect(alloc, epoch);
        tail = cown;
        cown = n;
      }
      if (tail != nullptr)
        add_cowns(head, tail);
    }

    void scan()
    {
      T* head = list.exchange(nullptr);
      T* tail = head;
      T* p = head;
      while (p != nullptr)
      {
        if (p->can_lifo_schedule())
          p->reschedule();
        tail = p;
        p = p->next;
      }
      if (tail != nullptr)
        add_cowns(head, tail);
    }

    /**
     * Atomically add a single cown to the list.
     */
    void add_cown(T* cown)
    {
      cown->next = list;
      while (!list.compare_exchange_weak(cown->next, cown))
      {
        cown->next = list;
      }
    }

    /*
     * Atomically add an entire list to the core list.
     * l*/
    void add_cowns(T* head, T* tail)
    {
      assert(head != nullptr && tail != nullptr);
      tail->next = list;
      while (!list.compare_exchange_weak(tail->next, head))
      {
        tail->next = list;
      }
    }

    /*
     * Take ownership of the entire content of the list.
     * */
    T* drain()
    {
      return list.exchange(nullptr);
    }
  };
}
