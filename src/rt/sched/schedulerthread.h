// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "ds/hashmap.h"
#include "ds/mpscq.h"
#include "mpmcq.h"
#include "object/object.h"
#include "priority.h"
#include "schedulerstats.h"
#include "threadpool.h"

#include <snmalloc.h>

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
    size_t systematic_speed_mask = 1;

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
    /// Used by systematic testing to implement the condition variable,
    /// and thread termination.
    SystematicState systematic_state = SystematicState::Active;

    /// Used to specify a condition when this thread should/could make
    /// progress.  It is used to implement condition variables.
    snmalloc::function_ref<bool()> guard = true_thunk;

    /// How many uninterrupted steps this threads has been selected to run for.
    size_t steps = 0;
#endif

    MPMCQ<T> q;
    Alloc* alloc = nullptr;
    SchedulerThread<T>* next = nullptr;
    SchedulerThread<T>* victim = nullptr;
    std::condition_variable cv;

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
    /// The mutor is the first high priority cown that receives a message from a
    /// set of cowns running a behaviour on this scheduler thread.
    T* mutor = nullptr;
    /// The set of cowns muted on this scheduler thread. These are unmuted and
    /// cleared before scheduler sleep, or in some stages of the LD protocol.
    ObjectMap<T*> mute_set;

    T* get_token_cown()
    {
      assert(token_cown);
      return token_cown;
    }

    SchedulerThread()
    : token_cown{T::create_token_cown()},
      q{token_cown},
      mute_set{ThreadAlloc::get()}
    {
      token_cown->set_owning_thread(this);
    }

    ~SchedulerThread()
    {
      assert(mute_set.size() == 0);
    }

    inline void stop()
    {
      running = false;
    }

    inline void schedule_fifo(T* a)
    {
      Systematic::cout() << "Enqueue cown " << a << " (" << a->get_epoch_mark()
                         << ")" << Systematic::endl;

      // Scheduling on this thread, from this thread.
      if (!a->scanned(send_epoch))
      {
        Systematic::cout() << "Enqueue unscanned cown " << a
                           << Systematic::endl;
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
      Systematic::cout() << "LIFO scheduling cown " << a << " onto "
                         << systematic_id << Systematic::endl;
      q.enqueue_front(ThreadAlloc::get(), a);
      Systematic::cout() << "LIFO scheduled cown " << a << " onto "
                         << systematic_id << Systematic::endl;

      stats.lifo();

      if (Scheduler::get().unpause())
        stats.unpause();
    }

    /**
     * Track a cown muted on this thread so that it may be unmuted prior to
     * shutdown.
     */
    void mute_set_add(T* cown)
    {
      bool inserted = mute_set.insert(*alloc, cown).first;
      if (inserted)
        cown->weak_acquire();
    }

    /**
     * Clear the mute set and unmute any muted cowns in the set.
     */
    void mute_set_clear()
    {
      Systematic::cout() << "Clear mute set" << Systematic::endl;
      for (auto entry = mute_set.begin(); entry != mute_set.end(); ++entry)
      {
        // This operation should be safe if the cown has been collected but the
        // stub exists.
        entry.key()->backpressure_transition(Priority::Normal);
        entry.key()->weak_release(*alloc);
      }
      mute_set.clear(*alloc);
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

      Scheduler::get().sync.thread_start(this);

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
            Systematic::cout()
              << "Pop cown " << clear_thread_bit(cown) << Systematic::endl;
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

        Systematic::cout() << "Schedule cown " << cown << " ("
                           << cown->get_epoch_mark() << ")" << Systematic::endl;

        // This prevents the LD protocol advancing if this cown has not been
        // scanned. This catches various cases where we have stolen, or
        // reschedule with the empty queue. We are effectively rescheduling, so
        // check if unscanned. This seems a little agressive, but prevents the
        // protocol advancing too quickly.
        // TODO refactor this could be made more optimal if we only do this for
        // stealing, and running on same cown as previous loop.
        if (Scheduler::should_scan() && (cown->get_epoch_mark() != send_epoch))
        {
          Systematic::cout() << "Unscanned cown next" << Systematic::endl;
          scheduled_unscanned_cown = true;
        }

        ld_protocol();

        Systematic::cout() << "Running cown " << cown << Systematic::endl;

        bool reschedule = cown->run(*alloc, state, send_epoch);

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
                Systematic::cout() << "Queue empty" << Systematic::endl;
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
                Systematic::cout()
                  << "Reschedule cown " << cown << " ("
                  << cown->get_epoch_mark() << ")" << Systematic::endl;
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

      assert(mute_set.size() == 0);

      Systematic::cout() << "Begin teardown (phase 1)" << Systematic::endl;

      cown = list;
      while (cown != nullptr)
      {
        if (!cown->is_collected())
          cown->collect(*alloc);
        cown = cown->next;
      }

      Systematic::cout() << "End teardown (phase 1)" << Systematic::endl;

      Epoch(ThreadAlloc::get()).flush_local();
      Scheduler::get().enter_barrier();

      Systematic::cout() << "Begin teardown (phase 2)" << Systematic::endl;

      GlobalEpoch::advance();

      collect_cown_stubs<true>();

      Systematic::cout() << "End teardown (phase 2)" << Systematic::endl;

      q.destroy(*alloc);

      Scheduler::get().sync.thread_finished(this);

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
          Systematic::cout()
            << "Fast-steal cown " << clear_thread_bit(cown) << " from "
            << victim->systematic_id << Systematic::endl;
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
      Systematic::cout() << "Reached LD token" << Systematic::endl;
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
            Systematic::cout()
              << "Stole cown " << clear_thread_bit(cown) << " from "
              << victim->systematic_id << Systematic::endl;
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

        if (mute_set.size() != 0)
        {
          mute_set_clear();
          continue;
        }

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
            Systematic::cout()
              << "Should steal for fairness!" << Systematic::endl;
            should_steal_for_fairness = true;
          }

          if (n_ld_tokens > 0)
          {
            dec_n_ld_tokens();
          }

          Systematic::cout() << "Reached token" << Systematic::endl;
        }
        else
        {
          Systematic::cout() << "Reached token: stolen from "
                             << sched->systematic_id << Systematic::endl;
        }

        // Put back the token
        sched->q.enqueue(*alloc, cown);
        return false;
      }

      // Register this cown with the scheduler thread if it is not currently
      // registered with a scheduler thread.
      if (cown->owning_thread() == nullptr)
      {
        Systematic::cout() << "Bind cown to scheduler thread: " << this
                           << Systematic::endl;
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
        Systematic::cout() << "==============================================="
                           << Systematic::endl;
        Systematic::cout() << "==============================================="
                           << Systematic::endl;
        Systematic::cout() << "==============================================="
                           << Systematic::endl;
        Systematic::cout() << "==============================================="
                           << Systematic::endl;

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
        Systematic::cout() << "Scheduler unscanned flag: "
                           << scheduled_unscanned_cown << Systematic::endl;

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
          Systematic::cout() << "LD protocol loop" << Systematic::endl;
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
      Systematic::cout() << "Scheduler state change: " << state << " -> "
                         << snext << Systematic::endl;
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

      Systematic::cout() << "send_epoch (1): " << send_epoch
                         << Systematic::endl;
    }

    void enter_scan()
    {
      send_epoch = (prev_epoch == EpochMark::EPOCH_B) ? EpochMark::EPOCH_A :
                                                        EpochMark::EPOCH_B;
      Systematic::cout() << "send_epoch (2): " << send_epoch
                         << Systematic::endl;

      // Send empty messages to all cowns that can be LIFO scheduled.

      mute_set_clear(); // TODO: is this necesary?

      T* p = list;
      while (p != nullptr)
      {
        if (p->can_lifo_schedule())
          p->reschedule();

        p = p->next;
      }

      n_ld_tokens = 2;
      scheduled_unscanned_cown = false;
      Systematic::cout() << "Enqueued LD check point" << Systematic::endl;
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
            Systematic::cout() << "Leaking cown " << c << Systematic::endl;
            if (Scheduler::get_detect_leaks())
            {
              *p = c->next;
              continue;
            }
          }
          Systematic::cout() << "Stub collect cown " << c << Systematic::endl;
          // TODO: Investigate systematic testing coverage here.
          auto epoch = c->epoch_when_popped;
          auto outdated =
            epoch == T::NO_EPOCH_SET || GlobalEpoch::is_outdated(epoch);
          if (outdated)
          {
            count++;
            *p = c->next;
            Systematic::cout()
              << "Stub collected cown " << c << Systematic::endl;
            c->dealloc(*alloc);
            continue;
          }
          else
          {
            if (!outdated)
              Systematic::cout()
                << "Cown " << c << " not outdated." << Systematic::endl;
          }
        }
        p = &(c->next);
      }

      free_cowns -= count;
    }
  };
} // namespace verona::rt
