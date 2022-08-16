// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "../pal/threadpoolbuilder.h"
#include "test/logging.h"
#include "threadstate.h"
#ifdef USE_SYSTEMATIC_TESTING
#  include "threadsyncsystematic.h"
#else
#  include "threadsync.h"
#endif

#include "corepool.h"
#include "schedulerlist.h"

#include <condition_variable>
#include <mutex>
#include <snmalloc/snmalloc.h>

namespace verona::rt
{
  /// Used for default prerun for a thread.
  inline void nop() {}

  using namespace snmalloc;
  template<class T, class E>
  class ThreadPool
  {
  private:
    friend T;
    friend void verona::rt::yield();

    static constexpr uint64_t TSC_PAUSE_SLOP = 1'000'000;
    static constexpr uint64_t TSC_UNPAUSE_SLOP = TSC_PAUSE_SLOP / 2;

    bool detect_leaks = true;
    size_t incarnation = 1;

    /**
     * Number of messages that have been sent that may not be visible to a
     *thread in a Scan state.
     **/
    std::atomic<size_t> inflight_count = 0;

    /**
     * Used to represent the current pause_epoch.
     *
     * If a thread is paused, then it must be the case
     * that pause_epoch is ahead of unpause_epoch.
     */
    std::atomic<uint64_t> pause_epoch{0};

    /**
     * Used to track unpause calls.  Threads unpausing
     * attempt to catch unpause_epoch up to pause_epoch,
     * and thus ensure threads are running.
     */
    std::atomic<uint64_t> unpause_epoch{0};

#ifdef USE_SYSTEMATIC_TESTING
    ThreadSyncSystematic<T> sync;
#else
    ThreadSync<T> sync;
#endif

    uint64_t barrier_incarnation = 0;

    /// List of instantiated scheduler threads.
    /// Contains both free and active threads; protects accesses with a lock.
    SchedulerList<T>* threads = nullptr;

    /// How many threads are being managed by this pool
    size_t thread_count = 0;

    /// Count of external event sources, such as I/O, that will prevent
    /// quiescence.
    size_t external_event_sources = 0;

    bool teardown_in_progress = false;

    bool fair = false;

    ThreadState state;

    /// Pool of cores shared by the scheduler threads.
    CorePool<ThreadPool<T, E>, E>* core_pool = nullptr;

    /// Systematic ids.
    size_t systematic_ids = 0;

  public:
    static ThreadPool<T, E>& get()
    {
      SNMALLOC_REQUIRE_CONSTINIT static ThreadPool<T, E> global_thread_pool;
      return global_thread_pool;
    }

    static Core<E>* first_core()
    {
      return get().core_pool->first_core;
    }

    static void set_detect_leaks(bool b)
    {
      get().detect_leaks = b;
    }

    static bool get_detect_leaks()
    {
      return get().detect_leaks;
    }

    static void record_inflight_message()
    {
      Logging::cout() << "Increase inflight count: " << get().inflight_count + 1
                      << Logging::endl;
      local()->scheduled_unscanned_cown = true;
      get().inflight_count++;
    }

    static void recv_inflight_message()
    {
      Logging::cout() << "Decrease inflight count: " << get().inflight_count - 1
                      << Logging::endl;
      get().inflight_count--;
    }

    static bool no_inflight_messages()
    {
      Logging::cout() << "Check inflight count: " << get().inflight_count
                      << Logging::endl;
      return get().inflight_count == 0;
    }

    /// Increment the external event source count. A non-zero count will prevent
    /// runtime teardown.
    /// This should only be called from inside the runtime.
    /// A message can be enqueued before the runtime is running if there is a
    /// external event source from the start.
    static void add_external_event_source()
    {
      auto& s = get();
      auto h = s.sync.handle(local());
      assert(local() != nullptr);
      auto prev_count = s.external_event_sources++;
      Logging::cout() << "Add external event source (now " << (prev_count + 1)
                      << ")" << Logging::endl;
    }

    /// Decrement the external event source count. This will allow runtime
    /// teardown if the count drops to zero.
    static void remove_external_event_source()
    {
      // Must be called from inside a scheduler thread,
      // A message can be LIFO scheduled to call this.
      // Note, if this is not called from a scheduler thread, then the following
      // can happen
      //   1. All scheduler threads decide to pause the runtime but stop just
      //      before waiting on the condition variable.
      //   2. This code runs and calls notify, but no threads see this.
      //   3. All the scheduler threads go call wait on condition variables.
      // This leads to the system becoming inactive, and will never wake up.
      // Forcing this code to be injected onto a scheduler thread by a message
      // means the runtime cannot be attempting to pause while this code is
      // running.
      assert(local() != nullptr);

      auto& s = get();
      auto h = s.sync.handle(local());
      auto prev_count = s.external_event_sources--;
      assert(prev_count != 0);
      Logging::cout() << "Remove external event source (now "
                      << (prev_count - 1) << ")" << Logging::endl;
    }

    static void set_fair(bool fair)
    {
      Logging::cout() << "Set fair: " << fair << Logging::endl;
      auto& s = get();
      s.fair = fair;
    }

    static bool is_teardown_in_progress()
    {
      return get().teardown_in_progress;
    }

    static T*& local()
    {
      static thread_local T* local;
      return local;
    }

    static Core<E>* round_robin()
    {
      static thread_local size_t incarnation;
      static thread_local Core<E>* nonlocal;

      if (incarnation != get().incarnation)
      {
        incarnation = get().incarnation;
        nonlocal = get().first_core();
      }
      else
      {
        nonlocal = nonlocal->next;
      }

      return nonlocal;
    }

    static EpochMark epoch()
    {
      T* t = local();

      if (t != nullptr)
        return t->send_epoch;

      return EpochMark::EPOCH_A;
    }

    static EpochMark alloc_epoch()
    {
      T* t = local();

      // TODO Review what epoch should external participants use?
      if (t == nullptr)
        return epoch();

      if (in_prescan())
      {
        // During pre-scan alloc in previous epoch.
        Logging::cout() << "Alloc cown during pre-scan" << Logging::endl;
        return t->prev_epoch;
      }

      return epoch();
    }

    static bool should_scan()
    {
      T* t = local();

      if (t == nullptr)
        return false;

      switch (t->state)
      {
        case ThreadState::Scan:
        case ThreadState::AllInScan:
        case ThreadState::BelieveDone_Voted:
        case ThreadState::BelieveDone:
        case ThreadState::BelieveDone_Confirm:
        case ThreadState::BelieveDone_Retract:
        case ThreadState::BelieveDone_Ack:
        case ThreadState::ReallyDone:
        case ThreadState::ReallyDone_Retract:
          return true;
        default:
          return false;
      }
    }

    static bool in_prescan()
    {
      T* t = local();

      if (t == nullptr)
        return false;

      return (t->state) == ThreadState::PreScan;
    }

    /**
     * Either the local or the global state is in prescan.  This should
     * only be used in assertions.
     **/
    static bool debug_in_prescan()
    {
      T* t = local();

      if (t == nullptr)
        return false;

      return ((t->state) == ThreadState::PreScan) ||
        ((get().state.get_state()) == ThreadState::PreScan);
    }

    static void want_ld()
    {
      T* t = local();

      if (t != nullptr)
        t->want_ld();
    }

    void init(size_t count)
    {
      Logging::cout() << "Init runtime" << Logging::endl;

      if ((thread_count != 0) || (count == 0))
        abort();

      thread_count = count;

      // Build a circular linked list of scheduler threads.
      threads = new SchedulerList<T>();
      teardown_in_progress = false;

      // Initialize the corepool.
      core_pool = new CorePool<ThreadPool<T, E>, E>(count);

      // For future ids.
      systematic_ids = count + 1;
      
      for (; count > 0; count--) {
        T* t = new T;
        t->systematic_id = count;
#ifdef USE_SYSTEMATIC_TESTING
        t->local_systematic =
          Systematic::create_systematic_thread(t->systematic_id);
#endif
        threads->addFree(t);
      }
      Logging::cout() << "Runtime initialised" << Logging::endl;
      init_barrier();
    }

    void run()
    {
      run_with_startup<>(&nop);
    }

    template<typename... Args>
    void run_with_startup(void (*startup)(Args...), Args... args)
    {
      {
        ThreadPoolBuilder builder(thread_count);

        Logging::cout() << "Starting all threads" << Logging::endl;
        for (size_t i = 0; i < thread_count; i++) 
        {
          T* t = threads->popFree();
          if (t == nullptr)
            abort();
          t->setCore(core_pool->cores[i]);
          threads->addActive(t);
          builder.add_thread(t->core->affinity, &T::run, t, startup, args...);
        }
      }
      Logging::cout() << "All threads stopped" << Logging::endl;

      T* t = threads->popActive();
      while(t != nullptr)
      {
        T* prev = t;
        t = threads->popActive();
        delete prev;
      } 
      t = threads->popFree();
      while (t != nullptr)
      {
        T* prev = t;
        t = threads->popFree();
        delete prev;
      }
      Logging::cout() << "All threads deallocated" << Logging::endl;

      incarnation++;
#ifdef USE_SYSTEMATIC_TESTING
      Object::reset_ids();
#endif
      thread_count = 0;
      state.reset<ThreadState::NotInLD>();

      Epoch::flush(ThreadAlloc::get());
      delete core_pool;
      delete threads;
    }

    static bool debug_not_running()
    {
      return get().state.get_active_threads() == 0;
    }

  private:
    inline ThreadState::State next_state(ThreadState::State s)
    {
      auto h = sync.handle(local());
      return state.next(s, thread_count);
    }

    bool check_for_work()
    {
      // TODO: check for pending async IO
      Core<E>* c = first_core();
      do
      {
        Logging::cout() << "Checking for pending work on thread "
                        << c->affinity << Logging::endl;
        if (!c->q.nothing_old())
        {
          Logging::cout() << "Found pending work!" << Logging::endl;
          return true;
        }
        c = c->next;
      } while (c != first_core());

      Logging::cout() << "No pending work!" << Logging::endl;
      return false;
    }

    bool pause()
    {
      // Snapshot unpause_epoch, so we can detect a racing unpause.
      auto local_unpause_epoch = unpause_epoch.load(std::memory_order_relaxed);

      yield();

      // Notify that we are trying to pause this thread.
      pause_epoch++;

      yield();

      // Strong barrier to ensure that this is visible to all threads before
      // we actually attempt to sleep.
#ifndef USE_SYSTEMATIC_TESTING
      // This has no effect as execution is sequentialised with systematic
      // testing. and causes bad performance.
      Barrier::memory();
#endif

      // Work has become available, we shouldn't pause.
      if (check_for_work())
        return false;

      yield();

      {
        auto h = sync.handle(local());

        // An unpause has occurred since, we started to pause.
        if (
          local_unpause_epoch != unpause_epoch.load(std::memory_order_relaxed))
          return false;

        // Check if we should wait for other threads to generate more work.
        auto value = state.get_active_threads();
        if (value > 1)
        {
          state.dec_active_threads();
          Logging::cout() << "Pausing" << Logging::endl;
          h.pause(); // Spurious wake-ups are safe.
          Logging::cout() << "Unpausing" << Logging::endl;
          state.inc_active_threads();
          return true;
        }

        // There are external sources should wait for external wake ups.
        if (external_event_sources != 0)
        {
          Logging::cout() << "Pausing last thread" << Logging::endl;
          h.pause(); // Spurious wake-ups are safe.
          Logging::cout() << "Unpausing last thread" << Logging::endl;
          return true;
        }

        Logging::cout() << "Teardown beginning" << Logging::endl;
        // Used to handle deallocating all the state of the threads.
        teardown_in_progress = true;

        // Tell all threads to stop looking for work.
        threads->forall([](T* thread){thread->stop();}); 
        Logging::cout() << "Teardown: all threads stopped" << Logging::endl;

        h.unpause_all();
        Logging::cout() << "cv_notify_all() for teardown" << Logging::endl;
      }
      Logging::cout() << "Teardown: all threads beginning teardown"
                      << Logging::endl;
      return true;
    }

    bool unpause()
    {
      Logging::cout() << "unpause()" << Logging::endl;

      // Work should be added before checking for the runtime_pause.
      Barrier::compiler();

      // The order of these loads does not mater.
      // They have been placed in the least helpful order to flush out bugs.
      auto local_pause_epoch = pause_epoch.load(std::memory_order_relaxed);
      yield();
      auto local_unpause_epoch = unpause_epoch.load(std::memory_order_relaxed);

      // Exit early if we think no threads are trying to sleep.
      // Our work will be visible to any thread at this point.
      if (local_unpause_epoch == local_pause_epoch)
        return false;

      yield();

      // Ensure our reading of pause_epoch occurred after
      // unpaused_epoch.  This is required to ensure we are going to
      // monotonically increase unpause_epoch.
      local_pause_epoch = pause_epoch.load(std::memory_order_acquire);

      yield();

      // Attempt to catch up epoch.
      bool success = unpause_epoch.compare_exchange_strong(
        local_unpause_epoch, local_pause_epoch);

      yield();

      if (success)
      {
        // This grabs the scheduler lock to ensure threads have seen CAS before
        // we notify.
        Logging::cout() << "Wake all threads" << Logging::endl;
        sync.unpause_all(local());
        return true;
      }
      // Another thread won the CAS race, and is responsible for waking up.
      return false;
    }

    void init_barrier()
    {
      state.set_barrier(thread_count);
    }

    void enter_barrier()
    {
      auto inc = barrier_incarnation;
      {
        auto h = sync.handle(local());
        auto barrier_count = state.exit_thread();
        if (barrier_count != 0)
        {
          while (inc == barrier_incarnation)
          {
            h.pause();
          }
          return;
        }
        init_barrier();
        barrier_incarnation++;
        h.unpause_all();
      }
    }
  };
} // namespace verona::rt
