// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "core.h"
#include "ds/dllist.h"
#include "ds/hashmap.h"
#include "ds/mpscq.h"
#include "mpmcq.h"
#include "object/object.h"
#include "schedulerlist.h"
#include "schedulerstats.h"
#include "threadpool.h"

#include <snmalloc/snmalloc.h>

namespace verona::rt
{
  /**
   * There is typically one scheduler thread pinned to each physical CPU core.
   * Each scheduler thread is responsible for running cowns in its queue and
   * periodically stealing cowns from the queues of other scheduler threads.
   * This periodic work stealing is done to fairly distribute work across the
   * available scheduler threads. The period of work stealing for fairness is
   * determined by a single token cown that will be dequeued once all cowns
   * before it have been run. The removal of the token cown from the queue
   * occurs at a rate inversely proportional to the amount of cowns pending work
   * on that thread. A scheduler thread will enqueue a new token, if its
   * previous one has been dequeued or stolen, once more work is scheduled on
   * the scheduler thread.
   */
  template<class T>
  class SchedulerThread
  {
  public:
    /// Friendly thread identifier for logging information.
    size_t systematic_id = 0;

  private:
    using Scheduler = ThreadPool<SchedulerThread<T>, T>;
    friend Scheduler;
    friend T;
    friend DLList<SchedulerThread<T>>;
    friend SchedulerList<SchedulerThread<T>>;

    template<typename Owner>
    friend class Noticeboard;

    static constexpr uint64_t TSC_QUIESCENCE_TIMEOUT = 1'000'000;

    Core<T>* core = nullptr;
#ifdef USE_SYSTEMATIC_TESTING
    friend class ThreadSyncSystematic<SchedulerThread>;
    Systematic::Local* local_systematic{nullptr};
#else
    friend class ThreadSync<SchedulerThread>;
    LocalSync local_sync{};
#endif

    Alloc* alloc = nullptr;
    Core<T>* victim = nullptr;

    bool running = true;

    bool should_steal_for_fairness = false;

    std::atomic<bool> scheduled_unscanned_cown = false;

    /// The MessageBody of a running behaviour.
    typename T::MessageBody* message_body = nullptr;

    /// SchedulerList pointers.
    SchedulerThread<T>* prev = nullptr;
    SchedulerThread<T>* next = nullptr;

    SchedulerThread()
    {
      Logging::cout() << "Scheduler Thread created" << Logging::endl;
    }

    ~SchedulerThread() {}

    void set_core(Core<T>* core)
    {
      this->core = core;
    }

    inline void stop()
    {
      running = false;
    }

    inline void schedule_fifo(T* a)
    {
      Logging::cout() << "Enqueue cown " << a << Logging::endl;

      assert(!a->queue.is_sleeping());
      core->q.enqueue(*alloc, a);

      if (Scheduler::get().unpause())
        core->stats.unpause();
    }

    static inline void schedule_lifo(Core<T>* c, T* a)
    {
      // A lifo scheduled cown is coming from an external source, such as
      // asynchronous I/O.
      Logging::cout() << "LIFO scheduling cown " << a << " onto " << c->affinity
                      << Logging::endl;
      c->q.enqueue_front(ThreadAlloc::get(), a);
      Logging::cout() << "LIFO scheduled cown " << a << " onto " << c->affinity
                      << Logging::endl;

      c->stats.lifo();

      if (Scheduler::get().unpause())
        c->stats.unpause();
    }

    template<typename... Args>
    static void run(SchedulerThread* t, void (*startup)(Args...), Args... args)
    {
      t->run_inner(startup, args...);
    }

    /**
     * Startup is supplied to initialise thread local state before the runtime
     * starts.
     *
     * This is used for initialising the interpreters per-thread data-structures
     **/
    template<typename... Args>
    void run_inner(void (*startup)(Args...), Args... args)
    {
      startup(args...);

      Scheduler::local() = this;
      alloc = &ThreadAlloc::get();
      assert(core != nullptr);
      victim = core->next;
      T* cown = nullptr;
      core->servicing_threads++;

#ifdef USE_SYSTEMATIC_TESTING
      Systematic::attach_systematic_thread(local_systematic);
#endif

      while (true)
      {
        if (should_steal_for_fairness)
        {
          if (cown == nullptr)
          {
            should_steal_for_fairness = false;
            fast_steal(cown);
          }
        }

        if (cown == nullptr)
        {
          cown = core->q.dequeue(*alloc);
          if (cown != nullptr)
            Logging::cout()
              << "Pop cown " << clear_thread_bit(cown) << Logging::endl;
        }

        if (cown == nullptr)
        {
          cown = steal();

          // If we can't steal, we are done.
          if (cown == nullptr)
            break;
        }

        // Administrative work before handling messages.
        if (!prerun(cown))
        {
          cown = nullptr;
          continue;
        }

        Logging::cout() << "Schedule cown " << cown << Logging::endl;

        core->progress_counter++;
        core->last_worker = systematic_id;

        bool reschedule = cown->run(*alloc);

        if (reschedule)
        {
          if (should_steal_for_fairness)
          {
            schedule_fifo(cown);
            cown = nullptr;
          }
          else
          {
            assert(!cown->queue.is_sleeping());
            // Push to the back of the queue if the queue is not empty,
            // otherwise run this cown again. Don't push to the queue
            // immediately to avoid another thread stealing our only cown.

            T* n = core->q.dequeue(*alloc);

            if (n != nullptr)
            {
              schedule_fifo(cown);
              cown = n;
            }
            else
            {
              if (core->q.nothing_old())
              {
                Logging::cout() << "Queue empty" << Logging::endl;
                // We have effectively reached token cown.

                T* stolen;
                if (Scheduler::get().fair && fast_steal(stolen))
                {
                  schedule_fifo(cown);
                  cown = stolen;
                }
              }

              if (!has_thread_bit(cown))
              {
                Logging::cout() << "Reschedule cown " << cown << Logging::endl;
              }
            }
          }
        }
        else
        {
          // Don't reschedule.
          cown = nullptr;
        }

        yield();
      }

      if (core != nullptr)
      {
        auto val = core->servicing_threads.fetch_sub(1);
        if (val == 1)
        {
          Logging::cout() << "Destroying core " << core->affinity
                          << Logging::endl;
          core->q.destroy(*alloc);
        }
      }

      Systematic::finished_thread();

      // Reset the local thread pointer as this physical thread could be reused
      // for a different SchedulerThread later.
      Scheduler::local() = nullptr;
    }

    bool fast_steal(T*& result)
    {
      // auto cur_victim = victim;
      T* cown;

      // Try to steal from the victim thread.
      if (victim != core)
      {
        cown = victim->q.dequeue(*alloc);

        if (cown != nullptr)
        {
          // stats.steal();
          Logging::cout() << "Fast-steal cown " << clear_thread_bit(cown)
                          << " from " << victim->affinity << Logging::endl;
          result = cown;
          return true;
        }
      }

      // We were unable to steal, move to the next victim thread.
      victim = victim->next;

      return false;
    }

    T* steal()
    {
      uint64_t tsc = Aal::tick();
      T* cown;

      while (running)
      {
        yield();

        // Check if some other thread has pushed work on our queue.
        cown = core->q.dequeue(*alloc);

        if (cown != nullptr)
          return cown;

        // Try to steal from the victim thread.
        if (victim != core)
        {
          cown = victim->q.dequeue(*alloc);

          if (cown != nullptr)
          {
            core->stats.steal();
            Logging::cout() << "Stole cown " << clear_thread_bit(cown)
                            << " from " << victim->affinity << Logging::endl;
            return cown;
          }
        }

        // We were unable to steal, move to the next victim thread.
        victim = victim->next;

#ifdef USE_SYSTEMATIC_TESTING
        // Only try to pause with 1/(2^5) probability
        UNUSED(tsc);
        if (!Systematic::coin(5))
        {
          yield();
          continue;
        }
#else
        // Wait until a minimum timeout has passed.
        uint64_t tsc2 = Aal::tick();
        if ((tsc2 - tsc) < TSC_QUIESCENCE_TIMEOUT)
        {
          Aal::pause();
          continue;
        }
#endif

        // We've been spinning looking for work for some time. While paused,
        // our running flag may be set to false, in which case we terminate.
        if (Scheduler::get().pause())
          core->stats.pause();
      }

      return nullptr;
    }

    bool has_thread_bit(T* cown)
    {
      return (uintptr_t)cown & 1;
    }

    T* clear_thread_bit(T* cown)
    {
      return (T*)((uintptr_t)cown & ~(uintptr_t)1);
    }

    /**
     * Some preliminaries required before we start processing messages
     *
     * - Check if this is the token, rather than a cown.
     *
     * This returns false, if this is a token, and true if it is real cown.
     **/
    bool prerun(T* cown)
    {
      // See if this is a SchedulerThread enqueued as a cown LD marker.
      // It may not be this one.
      if (has_thread_bit(cown))
      {
        auto unmasked = clear_thread_bit(cown);
        Core<T>* owning_core = unmasked->get_token_owning_core();

        if (owning_core == core)
        {
          if (Scheduler::get().fair)
          {
            Logging::cout() << "Should steal for fairness!" << Logging::endl;
            should_steal_for_fairness = true;
          }

          Logging::cout() << "Reached token" << Logging::endl;
        }
        else
        {
          Logging::cout() << "Reached token: stolen from "
                          << owning_core->affinity << Logging::endl;
        }

        // Put back the token
        owning_core->q.enqueue(*alloc, cown);
        return false;
      }

      return true;
    }
  };
} // namespace verona::rt
