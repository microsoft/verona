// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "ds/hashmap.h"
#include "ds/mpscq.h"
#include "mpmcq.h"
#include "object/object.h"
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
    using Scheduler = ThreadPool<SchedulerThread<T>>;
    friend Scheduler;
    friend T;

    template<typename Owner>
    friend class Noticeboard;

    static constexpr uint64_t TSC_QUIESCENCE_TIMEOUT = 1'000'000;

    T* token_cown = nullptr;

#ifdef USE_SYSTEMATIC_TESTING
    friend class ThreadSyncSystematic<SchedulerThread>;
    Systematic::Local* local_systematic{nullptr};
#else
    friend class ThreadSync<SchedulerThread>;
    LocalSync local_sync{};
#endif

    MPMCQ<T> q;
    Alloc* alloc = nullptr;
    SchedulerThread<T>* next = nullptr;
    SchedulerThread<T>* victim = nullptr;

    bool running = true;

    // `n_ld_tokens` indicates the times of token cown a scheduler has to
    // process before reaching its LD checkpoint (`n_ld_tokens == 0`).
    uint8_t n_ld_tokens = 0;

    bool should_steal_for_fairness = false;

    std::atomic<bool> scheduled_unscanned_cown = false;

    EpochMark send_epoch = EpochMark::EPOCH_A;
    EpochMark prev_epoch = EpochMark::EPOCH_B;

    ThreadState::State state = ThreadState::State::NotInLD;
    SchedulerStats stats;

    T* list = nullptr;
    size_t total_cowns = 0;
    std::atomic<size_t> free_cowns = 0;

    /// The MessageBody of a running behaviour.
    typename T::MessageBody* message_body = nullptr;

    T* get_token_cown()
    {
      assert(token_cown);
      return token_cown;
    }

    SchedulerThread() : token_cown{T::create_token_cown()}, q{token_cown}
    {
      token_cown->set_owning_thread(this);
    }

    ~SchedulerThread() {}

    inline void stop()
    {
      running = false;
    }

    inline void schedule_fifo(T* a)
    {
      Logging::cout() << "Enqueue cown " << a << " (" << a->get_epoch_mark()
                      << ")" << Logging::endl;

      // Scheduling on this thread, from this thread.
      if (!a->scanned(send_epoch))
      {
        Logging::cout() << "Enqueue unscanned cown " << a << Logging::endl;
        scheduled_unscanned_cown = true;
      }
      assert(!a->queue.is_sleeping());
      q.enqueue(*alloc, a);

      if (Scheduler::get().unpause())
        stats.unpause();
    }

    inline void schedule_lifo(T* a)
    {
      // A lifo scheduled cown is coming from an external source, such as
      // asynchronous I/O.
      Logging::cout() << "LIFO scheduling cown " << a << " onto "
                      << systematic_id << Logging::endl;
      q.enqueue_front(ThreadAlloc::get(), a);
      Logging::cout() << "LIFO scheduled cown " << a << " onto "
                      << systematic_id << Logging::endl;

      stats.lifo();

      if (Scheduler::get().unpause())
        stats.unpause();
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
      victim = next;
      T* cown = nullptr;

#ifdef USE_SYSTEMATIC_TESTING
      Systematic::attach_systematic_thread(this->local_systematic);
#endif

      while (true)
      {
        if (
          (total_cowns < (free_cowns << 1))
#ifdef USE_SYSTEMATIC_TESTING
          || Systematic::coin()
#endif
        )
          collect_cown_stubs();

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
          cown = q.dequeue(*alloc);
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

        Logging::cout() << "Schedule cown " << cown << " ("
                        << cown->get_epoch_mark() << ")" << Logging::endl;

        // This prevents the LD protocol advancing if this cown has not been
        // scanned. This catches various cases where we have stolen, or
        // reschedule with the empty queue. We are effectively rescheduling, so
        // check if unscanned. This seems a little agressive, but prevents the
        // protocol advancing too quickly.
        // TODO refactor this could be made more optimal if we only do this for
        // stealing, and running on same cown as previous loop.
        if (Scheduler::should_scan() && (cown->get_epoch_mark() != send_epoch))
        {
          Logging::cout() << "Unscanned cown next" << Logging::endl;
          scheduled_unscanned_cown = true;
        }

        ld_protocol();

        Logging::cout() << "Running cown " << cown << Logging::endl;

        bool reschedule = cown->run(*alloc, state);

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

            T* n = q.dequeue(*alloc);

            if (n != nullptr)
            {
              schedule_fifo(cown);
              cown = n;
            }
            else
            {
              if (q.nothing_old())
              {
                Logging::cout() << "Queue empty" << Logging::endl;
                // We have effectively reached token cown.
                n_ld_tokens = 0;

                T* stolen;
                if (Scheduler::get().fair && fast_steal(stolen))
                {
                  schedule_fifo(cown);
                  cown = stolen;
                }
              }

              if (!has_thread_bit(cown))
              {
                Logging::cout()
                  << "Reschedule cown " << cown << " ("
                  << cown->get_epoch_mark() << ")" << Logging::endl;
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

      Logging::cout() << "Begin teardown (phase 1)" << Logging::endl;

      cown = list;
      while (cown != nullptr)
      {
        if (!cown->is_collected())
          cown->collect(*alloc);
        cown = cown->next;
      }

      Logging::cout() << "End teardown (phase 1)" << Logging::endl;

      Epoch(ThreadAlloc::get()).flush_local();
      Scheduler::get().enter_barrier();

      Logging::cout() << "Begin teardown (phase 2)" << Logging::endl;

      GlobalEpoch::advance();

      collect_cown_stubs<true>();

      Logging::cout() << "End teardown (phase 2)" << Logging::endl;

      q.destroy(*alloc);

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
      if (victim != this)
      {
        cown = victim->q.dequeue(*alloc);

        if (cown != nullptr)
        {
          // stats.steal();
          Logging::cout() << "Fast-steal cown " << clear_thread_bit(cown)
                          << " from " << victim->systematic_id << Logging::endl;
          result = cown;
          return true;
        }
      }

      // We were unable to steal, move to the next victim thread.
      victim = victim->next;

      return false;
    }

    void dec_n_ld_tokens()
    {
      assert(n_ld_tokens == 1 || n_ld_tokens == 2);
      Logging::cout() << "Reached LD token" << Logging::endl;
      n_ld_tokens--;
    }

    T* steal()
    {
      uint64_t tsc = Aal::tick();
      T* cown;

      while (running)
      {
        yield();

        if (q.nothing_old())
        {
          n_ld_tokens = 0;
        }

        // Participate in the cown LD protocol.
        ld_protocol();

        // Check if some other thread has pushed work on our queue.
        cown = q.dequeue(*alloc);

        if (cown != nullptr)
          return cown;

        // Try to steal from the victim thread.
        if (victim != this)
        {
          cown = victim->q.dequeue(*alloc);

          if (cown != nullptr)
          {
            stats.steal();
            Logging::cout()
              << "Stole cown " << clear_thread_bit(cown) << " from "
              << victim->systematic_id << Logging::endl;
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

        // Enter sleep only if we aren't executing the leak detector currently.
        if (state == ThreadState::NotInLD)
        {
          // We've been spinning looking for work for some time. While paused,
          // our running flag may be set to false, in which case we terminate.
          if (Scheduler::get().pause())
            stats.pause();
        }
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
     * - Register cown to scheduler thread if not already on one.
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
        SchedulerThread* sched = unmasked->owning_thread();

        if (sched == this)
        {
          if (Scheduler::get().fair)
          {
            Logging::cout() << "Should steal for fairness!" << Logging::endl;
            should_steal_for_fairness = true;
          }

          if (n_ld_tokens > 0)
          {
            dec_n_ld_tokens();
          }

          Logging::cout() << "Reached token" << Logging::endl;
        }
        else
        {
          Logging::cout() << "Reached token: stolen from "
                          << sched->systematic_id << Logging::endl;
        }

        // Put back the token
        sched->q.enqueue(*alloc, cown);
        return false;
      }

      // Register this cown with the scheduler thread if it is not currently
      // registered with a scheduler thread.
      if (cown->owning_thread() == nullptr)
      {
        Logging::cout() << "Bind cown to scheduler thread: " << this
                        << Logging::endl;
        cown->set_owning_thread(this);
        cown->next = list;
        list = cown;
        total_cowns++;
      }

      return true;
    }

    void want_ld()
    {
      if (state == ThreadState::NotInLD)
      {
        Logging::cout() << "==============================================="
                        << Logging::endl;
        Logging::cout() << "==============================================="
                        << Logging::endl;
        Logging::cout() << "==============================================="
                        << Logging::endl;
        Logging::cout() << "==============================================="
                        << Logging::endl;

        ld_state_change(ThreadState::WantLD);
      }
    }

    bool ld_checkpoint_reached()
    {
      return n_ld_tokens == 0;
    }

    /**
     * This function updates the current thread state in the cown collection
     * protocol. This basically plays catch up with the global state, and can
     * vote for new states.
     **/
    void ld_protocol()
    {
      // Set state to BelieveDone_Vote when we think we've finished scanning.
      if ((state == ThreadState::AllInScan) && ld_checkpoint_reached())
      {
        Logging::cout() << "Scheduler unscanned flag: "
                        << scheduled_unscanned_cown << Logging::endl;

        if (!scheduled_unscanned_cown && Scheduler::no_inflight_messages())
        {
          ld_state_change(ThreadState::BelieveDone_Vote);
        }
        else
        {
          enter_scan();
        }
      }

      bool first = true;

      while (true)
      {
        ThreadState::State sprev = state;
        // Next state can affect global thread pool state, so add to testing for
        // systematic testing.
        yield();
        ThreadState::State snext = Scheduler::get().next_state(sprev);

        // If we have a lost wake-up, then all threads can get stuck
        // trying to perform a LD.
        if (
          sprev == ThreadState::PreScan && snext == ThreadState::PreScan &&
          Scheduler::get().unpause())
        {
          stats.unpause();
        }

        if (snext == sprev)
          return;
        yield();

        if (first)
        {
          first = false;
          Logging::cout() << "LD protocol loop" << Logging::endl;
        }

        ld_state_change(snext);

        // Actions taken when a state transition occurs.
        switch (state)
        {
          case ThreadState::PreScan:
          {
            if (Scheduler::get().unpause())
              stats.unpause();

            enter_prescan();
            return;
          }

          case ThreadState::Scan:
          {
            if (sprev != ThreadState::PreScan)
              enter_prescan();
            enter_scan();
            return;
          }

          case ThreadState::AllInScan:
          {
            if (sprev == ThreadState::PreScan)
              enter_scan();
            return;
          }

          case ThreadState::BelieveDone:
          {
            if (scheduled_unscanned_cown)
              ld_state_change(ThreadState::BelieveDone_Retract);
            else
              ld_state_change(ThreadState::BelieveDone_Confirm);
            continue;
          }

          case ThreadState::ReallyDone_Confirm:
          {
            continue;
          }

          case ThreadState::Sweep:
          {
            collect_cowns();
            continue;
          }

          default:
          {
            continue;
          }
        }
      }
    }

    bool in_sweep_state()
    {
      return state == ThreadState::Sweep;
    }

    void ld_state_change(ThreadState::State snext)
    {
      Logging::cout() << "Scheduler state change: " << state << " -> " << snext
                      << Logging::endl;
      state = snext;
    }

    void enter_prescan()
    {
      // Save epoch for when we start scanning
      prev_epoch = send_epoch;

      // Set sending Epoch to EpochNone. As these new messages need to be
      // counted to ensure all inflight work is processed before we complete
      // scanning.
      send_epoch = EpochMark::EPOCH_NONE;

      Logging::cout() << "send_epoch (1): " << send_epoch << Logging::endl;
    }

    void enter_scan()
    {
      send_epoch = (prev_epoch == EpochMark::EPOCH_B) ? EpochMark::EPOCH_A :
                                                        EpochMark::EPOCH_B;
      Logging::cout() << "send_epoch (2): " << send_epoch << Logging::endl;

      // Send empty messages to all cowns that can be LIFO scheduled.

      T* p = list;
      while (p != nullptr)
      {
        if (p->can_lifo_schedule())
          p->reschedule();

        p = p->next;
      }

      n_ld_tokens = 2;
      scheduled_unscanned_cown = false;
      Logging::cout() << "Enqueued LD check point" << Logging::endl;
    }

    void collect_cowns()
    {
      T* p = list;

      while (p != nullptr)
      {
        T* n = p->next;
        p->try_collect(*alloc, send_epoch);
        p = n;
      }
    }

    template<bool during_teardown = false>
    void collect_cown_stubs()
    {
      // Cannot collect the cown state while another thread could be
      // sweeping.  The other thread could be checking to see if it should
      // issue a decref to the object that is part of the same collection,
      // and thus cause a use-after-free.
      switch (state)
      {
        case ThreadState::ReallyDone_Confirm:
        case ThreadState::Finished:
          return;

        default:;
      }

      T** p = &list;
      size_t count = 0;

      while (*p != nullptr)
      {
        T* c = *p;
        // Collect cown stubs when the weak count is zero.
        if (c->weak_count == 0 || during_teardown)
        {
          if (c->weak_count != 0)
          {
            Logging::cout() << "Leaking cown " << c << Logging::endl;
            if (Scheduler::get_detect_leaks())
            {
              *p = c->next;
              continue;
            }
          }
          Logging::cout() << "Stub collect cown " << c << Logging::endl;
          // TODO: Investigate systematic testing coverage here.
          auto epoch = c->epoch_when_popped;
          auto outdated =
            epoch == T::NO_EPOCH_SET || GlobalEpoch::is_outdated(epoch);
          if (outdated)
          {
            count++;
            *p = c->next;
            Logging::cout() << "Stub collected cown " << c << Logging::endl;
            c->dealloc(*alloc);
            continue;
          }
          else
          {
            if (!outdated)
              Logging::cout()
                << "Cown " << c << " not outdated." << Logging::endl;
          }
        }
        p = &(c->next);
      }

      free_cowns -= count;
    }
  };
} // namespace verona::rt
