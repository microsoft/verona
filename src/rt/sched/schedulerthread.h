// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "cpu.h"
#include "ds/hashmap.h"
#include "ds/mpscq.h"
#include "object/object.h"
#include "schedulerstats.h"
#include "spmcq.h"
#include "status.h"
#include "threadpool.h"

#include <snmalloc.h>
#include <thread>

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

    static void yield()
    {
#ifdef USE_SYSTEMATIC_TESTING
      Scheduler::yield_my_turn();
#endif
    }

    template<typename Owner>
    friend class Noticeboard;

    static constexpr uint64_t TSC_QUIESCENCE_TIMEOUT = 1'000'000;

    T* token_cown = nullptr;

#ifdef USE_SYSTEMATIC_TESTING
    /// Used by systematic testing to implement the condition variable.
    /// If true, then this thread is being simulated to be a sleep waiting for
    /// an unpause of a thread.
    bool sleeping = false;
#endif

    SPMCQ<T> q;
    Alloc* alloc = nullptr;
    SchedulerThread<T>* next = nullptr;
    SchedulerThread<T>* victim = nullptr;
    std::condition_variable cv;

    bool running = true;

    // `n_ld_tokens` indicates the times of token cown a scheduler has to
    // process before reaching its LD checkpoint (`n_ld_tokens == 0`).
    uint8_t n_ld_tokens = 0;

    // The cown queue is initialized with only the token (a cown) in.
    // Whenever the token is popped out, `token_consumed` is set to `true`,
    // informing its owner so that it could re-insert the token, which is
    // required because there's always one cown stuck in the queue; if the
    // token is not there, this must mean a real cown is stuck there.
    // Accordingly, the `is_empty` returns true iff token is the only item
    // left in the queue.
    std::atomic<bool> token_consumed = false;
    bool should_steal_for_fairness = false;

    std::atomic<bool> scheduled_unscanned_cown = false;

    EpochMark send_epoch = EpochMark::EPOCH_A;
    EpochMark prev_epoch = EpochMark::EPOCH_B;
    size_t affinity = (size_t)-1;

    std::thread t;
    ThreadState::State state = ThreadState::State::NotInLD;
    SchedulerStats stats;

    T* list = nullptr;
    size_t total_cowns = 0;
    std::atomic<size_t> free_cowns = 0;

    ObjectMap<std::pair<T*, ObjectMap<T*>*>> mute_map;
    typename T::MessageBody* message_body = nullptr;
    T* mutor = nullptr;

    T* get_token_cown()
    {
      assert(token_cown);
      return token_cown;
    }

    SchedulerThread()
    : token_cown{T::create_token_cown()},
      q{token_cown},
      mute_map{ThreadAlloc::get()}
    {
      token_cown->set_owning_thread(this);
    }

    ~SchedulerThread()
    {
      if (t.joinable())
        t.join();

      assert(mute_map.size() == 0);
    }

    template<typename... Args>
    inline void start(size_t af, void (*startup)(Args...), Args... args)
    {
      affinity = af;
      t = std::thread(&SchedulerThread::run<Args...>, this, startup, args...);
    }

    inline void stop()
    {
      running = false;
    }

    inline void schedule_fifo(T* a)
    {
      Systematic::cout() << "Enqueue cown: " << a << " (" << a->get_epoch_mark()
                         << ")" << std::endl;

      // Scheduling on this thread, from this thread.
      if (!a->scanned(send_epoch))
      {
        Systematic::cout() << "Enqueue unscanned cown: " << a << std::endl;
        scheduled_unscanned_cown = true;
      }
      assert(!a->queue.is_sleeping());
      q.enqueue(alloc, a);

      // Put the token back if it has been stolen.  This will help
      // free up more work for other threads to steal.
      check_token_cown();

      if (Scheduler::get().unpause())
        stats.unpause();
    }

    inline void schedule_lifo(T* a)
    {
      // A lifo scheduled cown is coming from an external source, such as
      // asynchronous I/O.
      Systematic::cout() << "LIFO schedule cown: " << a << std::endl;

      q.enqueue_front(ThreadAlloc::get(), a);
      stats.lifo();

      if (Scheduler::get().unpause())
        stats.unpause();
    }

    void check_token_cown()
    {
      if (is_token_consumed())
      {
        Systematic::cout() << "Put token " << get_token_cown()
                           << " in scheduler queue." << std::endl;
        if (n_ld_tokens > 0)
        {
          dec_n_ld_tokens();
        }
        set_token_consumed(false);
        enqueue_token();

        if (Scheduler::get().fair)
        {
          Systematic::cout() << "Should steal for fairness!" << std::endl;
          should_steal_for_fairness = true;
        }
      }
    }

    /**
     * Mute a set of cowns. This will add the cowns to the mute set of the
     * mutor.
     */
    void mute(T** cowns, size_t count)
    {
      assert(mutor != nullptr);

      auto it = mute_map.find(mutor);
      if (it == mute_map.end())
      {
        auto* set = ObjectMap<T*>::create(alloc);
        it = mute_map.insert(alloc, std::make_pair(mutor, set)).second;
      }
      else
      {
        mutor->weak_release(alloc);
      }
      auto& mute_set = *it.value();

      for (size_t i = 0; i < count; i++)
      {
        auto* cown = cowns[i];
        auto bp = cown->bp_state.load(std::memory_order_relaxed);
        yield();
        assert(bp != BackpressureState::Muted);

        if (
          (bp & BackpressureState::IsUnmutable) ||
          (state == ThreadState::PreScan) || (state == ThreadState::Scan) ||
          (state == ThreadState::AllInScan))
        { // Messages in this cown's queue must be scanned.
          cown->schedule();
          continue;
        }

        yield();

        auto ins = mute_set.insert(alloc, cown);
        if (ins.first)
          T::acquire(cown);

        if (
#ifdef USE_SYSTEMATIC_TESTING
          Systematic::coin(9) ||
#endif
          !cown->bp_state.compare_exchange_weak(
            bp, BackpressureState::Muted, std::memory_order_acq_rel))
        {
          yield();
          assert(bp != BackpressureState::Muted);
          cown->schedule();
          mute_set.erase(ins.second);
          if (ins.first)
            T::release(alloc, cown);

          continue;
        }
        Systematic::cout() << "Cown " << this << ": backpressure state " << bp
                           << " -> Muted" << std::endl;
        assert(!(bp & BackpressureState::IsUnmutable));
      }

      alloc->dealloc(cowns, count * sizeof(T*));
    }

    /**
     * Unmute all mute sets where the mutor is in a state that triggers
     * unmuting. If `force` is true, then all mute sets in the map will be
     * unmuted.
     */
    void mute_map_scan(bool force = false)
    {
      for (auto entry = mute_map.begin(); entry != mute_map.end(); ++entry)
      {
        auto* m = entry.key();
        if (force || m->triggers_unmuting())
        {
          yield();
          auto& mute_set = *entry.value();
          for (auto it = mute_set.begin(); it != mute_set.end(); ++it)
          {
            Systematic::cout() << "Mute map remove " << it.key() << std::endl;
            it.key()->unmute();
            T::release(alloc, it.key());
            mute_set.erase(it);
          }
          m->weak_release(alloc);
          mute_map.erase(entry);
          mute_set.dealloc(alloc);
          alloc->dealloc<sizeof(ObjectMap<T*>)>(&mute_set);
        }
      }

      if (mute_map.size() == 0)
        mute_map.clear(alloc);
    }

    /**
     * Startup is supplied to initialise thread local state before the runtime
     * starts.
     *
     * This is used for initialising the interpreters per-thread data-structures
     **/
    template<typename... Args>
    void run(void (*startup)(Args...), Args... args)
    {
      startup(args...);
      // Don't use affinity with systematic testing.  We're only ever running
      // one thread at a time in systematic testing mode and by pinning each
      // thread to a core we massively increase contention.
#ifndef USE_SYSTEMATIC_TESTING
      cpu::set_affinity(affinity);
#endif

      Scheduler::local() = this;
      alloc = ThreadAlloc::get();
      victim = next;
      T* cown = nullptr;

#ifdef USE_SYSTEMATIC_TESTING
      Scheduler::wait_for_my_first_turn();
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

        check_token_cown();

        mute_map_scan();

        if (cown == nullptr)
        {
          cown = q.dequeue(alloc);
          if (cown != nullptr)
            Systematic::cout() << "Pop cown: " << cown << std::endl;
        }

        if (cown == nullptr)
        {
          cown = steal();

          // If we can't steal, we are done.
          if (cown == nullptr)
            break;
        }

        Systematic::cout() << "Schedule cown: " << cown << " ("
                           << cown->get_epoch_mark() << ")" << std::endl;

        // Administrative work before handling messages.
        if (!prerun(cown))
        {
          cown = nullptr;
          continue;
        }

        // This prevents the LD protocol advancing if this cown has not been
        // scanned. This catches various cases where we have stolen, or
        // reschedule with the empty queue. We are effectively rescheduling, so
        // check if unscanned. This seems a little agressive, but prevents the
        // protocol advancing too quickly.
        // TODO refactor this could be made more optimal if we only do this for
        // stealing, and running on same cown as previous loop.
        if (Scheduler::should_scan() && (cown->get_epoch_mark() != send_epoch))
        {
          Systematic::cout() << "Unscanned cown next" << std::endl;
          scheduled_unscanned_cown = true;
        }

        ld_protocol();

        Systematic::cout() << "Running cown: " << cown << std::endl;

        bool reschedule = cown->run(alloc, state, send_epoch);

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

            T* n = q.dequeue(alloc);

            if (n != nullptr)
            {
              schedule_fifo(cown);
              cown = n;
            }
            else
            {
              if (q.is_empty())
              {
                Systematic::cout() << "Queue empty" << std::endl;
                // We have effectively reached token cown.
                n_ld_tokens = 0;

                T* stolen;
                if (Scheduler::get().fair && fast_steal(stolen))
                {
                  schedule_fifo(cown);
                  cown = stolen;
                }
              }

              Systematic::cout() << "Reschedule cown: " << cown << " ("
                                 << cown->get_epoch_mark() << ")" << std::endl;
            }
          }
        }
        else
        {
          Systematic::cout() << "Unschedule cown: " << cown << std::endl;
          // Don't reschedule.
          cown = nullptr;
        }

#ifdef USE_SYSTEMATIC_TESTING
        Scheduler::yield_my_turn();
#endif
      }

      assert(mute_map.size() == 0);

      Systematic::cout() << "Begin teardown (phase 1)" << std::endl;

      cown = list;
      while (cown != nullptr)
      {
        if (!cown->is_collected())
          cown->collect(alloc);
        cown = cown->next;
      }

      Systematic::cout() << "End teardown (phase 1)" << std::endl;

      Epoch(ThreadAlloc::get()).flush_local();
      Scheduler::get().enter_barrier();

      Systematic::cout() << "Begin teardown (phase 2)" << std::endl;

      GlobalEpoch::advance();

      collect_cown_stubs<true>();

      Systematic::cout() << "End teardown (phase 2)" << std::endl;

      q.destroy(alloc);
    }

    bool fast_steal(T*& result)
    {
      // auto cur_victim = victim;
      T* cown;

      // Try to steal from the victim thread.
      if (victim != this)
      {
        cown = victim->q.dequeue(alloc);

        if (cown != nullptr)
        {
          // stats.steal();
          Systematic::cout() << "Fast-steal cown: " << cown << " from "
                             << victim->systematic_id << std::endl;
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
      Systematic::cout() << "Reached LD token" << std::endl;
      n_ld_tokens--;
    }

    bool is_token_consumed()
    {
      auto res = token_consumed.load(std::memory_order_relaxed);
      yield();
      return res;
    }

    bool debug_is_token_consumed()
    {
      auto res = token_consumed.load(std::memory_order_relaxed);
      return res;
    }

    void set_token_consumed(bool res)
    {
      yield();
      token_consumed.store(res, std::memory_order_relaxed);
    }

    T* steal()
    {
      uint64_t tsc = Aal::tick();
      T* cown;

      while (running)
      {
        check_token_cown();

        yield();

        if (q.is_empty())
        {
          n_ld_tokens = 0;
        }

        // Participate in the cown LD protocol.
        ld_protocol();

        // Check if some other thread has pushed work on our queue.
        cown = q.dequeue(alloc);

        if (cown != nullptr)
          return cown;

        // Try to steal from the victim thread.
        if (victim != this)
        {
          cown = victim->q.dequeue(alloc);

          if (cown != nullptr)
          {
            stats.steal();
            Systematic::cout() << "Stole cown: " << cown << " from "
                               << victim->systematic_id << std::endl;
            return cown;
          }
        }

        // We were unable to steal, move to the next victim thread.
        victim = victim->next;

        // Wait until a minimum timeout has passed.
        uint64_t tsc2 = Aal::tick();

#ifndef USE_SYSTEMATIC_TESTING
        if ((tsc2 - tsc) < TSC_QUIESCENCE_TIMEOUT)
        {
          Aal::pause();
        }
        else
#else
        {
          UNUSED(tsc);
        }
#endif
          if (mute_map.size() != 0)
        {
          mute_map_scan(true);
          continue;
        }
        // Enter sleep only when the queue doesn't contain any real cowns.
        else if (state == ThreadState::NotInLD && q.is_empty())
        {
          // We've been spinning looking for work for some time. While paused,
          // our running flag may be set to false, in which case we terminate.
          if (Scheduler::get().pause(tsc2))
            stats.pause();
        }
#ifdef USE_SYSTEMATIC_TESTING
        else
        {
          yield();
        }
#endif
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
      // See if this is a SchedulerThread enqueued as an cown LD marker.
      // It may not be this one.
      if (has_thread_bit(cown))
      {
        auto unmasked = clear_thread_bit(cown);
        SchedulerThread* sched = unmasked->owning_thread();
        assert(!sched->debug_is_token_consumed());
        sched->set_token_consumed(true);

        if (sched != this)
        {
          Systematic::cout() << "Reached token: stolen from "
                             << sched->systematic_id << std::endl;
        }
        else
        {
          Systematic::cout() << "Reached token" << std::endl;
        }

        return false;
      }

      // Register this cown with the scheduler thread if it is not currently
      // registered with a scheduler thread.
      if (cown->owning_thread() == nullptr)
      {
        Systematic::cout() << "Bind cown to scheduler thread: " << this
                           << std::endl;
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
                           << std::endl;
        Systematic::cout() << "==============================================="
                           << std::endl;
        Systematic::cout() << "==============================================="
                           << std::endl;
        Systematic::cout() << "==============================================="
                           << std::endl;

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
                           << scheduled_unscanned_cown << std::endl;

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
          Systematic::cout() << "LD protocol loop" << std::endl;
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
                         << snext << std::endl;
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

      Systematic::cout() << "send_epoch (1): " << send_epoch << std::endl;
    }

    void enqueue_token()
    {
      // Must set the flag before pushing due to work stealing.
      assert(!debug_is_token_consumed());
      q.enqueue(alloc, (T*)((uintptr_t)get_token_cown() | 1));
    }

    void enter_scan()
    {
      send_epoch = (prev_epoch == EpochMark::EPOCH_B) ? EpochMark::EPOCH_A :
                                                        EpochMark::EPOCH_B;
      Systematic::cout() << "send_epoch (2): " << send_epoch << std::endl;

      // Send empty messages to all cowns that can be LIFO scheduled.

      mute_map_scan(true);

      T* p = list;
      while (p != nullptr)
      {
        if (p->can_lifo_schedule())
          p->reschedule();

        p = p->next;
      }

      n_ld_tokens = 2;
      scheduled_unscanned_cown = false;
      Systematic::cout() << "Enqueued LD check point" << std::endl;
    }

    void collect_cowns()
    {
      T* p = list;

      while (p != nullptr)
      {
        T* n = p->next;
        p->try_collect(alloc, send_epoch);
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
            Systematic::cout() << "Leaking cown: " << c << std::endl;
            if (Scheduler::get_detect_leaks())
            {
              *p = c->next;
              continue;
            }
          }
          Systematic::cout() << "Stub collect: " << c << std::endl;
          // TODO: Investigate systematic testing coverage here.
          auto epoch = c->epoch_when_popped;
          auto outdated =
            epoch == T::NO_EPOCH_SET || GlobalEpoch::is_outdated(epoch);
          if (outdated)
          {
            count++;
            *p = c->next;
            c->dealloc(alloc);
            Systematic::cout() << "Stub collected: " << c << std::endl;
            continue;
          }
          else
          {
            if (!outdated)
              Systematic::cout()
                << "Cown " << c << " not outdated." << std::endl;
          }
        }
        p = &(c->next);
      }

      free_cowns -= count;
    }
  };
} // namespace verona::rt
