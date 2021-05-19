// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "../pal/threadpoolbuilder.h"
#include "test/systematic.h"
#include "threadstate.h"
#ifdef USE_SYSTEMATIC_TESTING
#  include "threadsyncsystematic.h"
#else
#  include "threadsync.h"
#endif

#include <condition_variable>
#include <mutex>
#include <snmalloc.h>

namespace verona::rt
{
  /// Used for default prerun for a thread.
  inline void nop() {}

  using namespace snmalloc;
  template<class T>
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

    uint64_t last_unpause_tsc = Aal::tick();

#ifdef USE_SYSTEMATIC_TESTING
    ThreadSyncSystematic<T> sync;
#else
    ThreadSync<T> sync;
#endif

    std::atomic_uint64_t barrier_count = 0;
    uint64_t barrier_incarnation = 0;

    T* first_thread = nullptr;

    /// How many threads are being managed by this pool
    size_t thread_count = 0;

    /// How many threads are not currently paused.
    size_t active_thread_count = 0;

    /// Count of external event sources, such as I/O, that will prevent
    /// quiescence.
    std::atomic<size_t> external_event_sources = 0;
    // Pausing if value is odd.
    // Is not atomic, since updates are only made while a lock is held.
    // We are assuming that no partial write will be observed.
    uint32_t runtime_pausing = 0;
    bool teardown_in_progress = false;

    bool fair = false;

    ThreadState state;

  public:
    static ThreadPool<T>& get()
    {
      static ThreadPool<T> global_thread_pool;
      return global_thread_pool;
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
      Systematic::cout() << "Increase inflight count: "
                         << get().inflight_count + 1 << Systematic::endl;
      local()->scheduled_unscanned_cown = true;
      get().inflight_count++;
    }

    static void recv_inflight_message()
    {
      Systematic::cout() << "Decrease inflight count: "
                         << get().inflight_count - 1 << Systematic::endl;
      get().inflight_count--;
    }

    static bool no_inflight_messages()
    {
      Systematic::cout() << "Check inflight count: " << get().inflight_count
                         << Systematic::endl;
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
      auto prev_count =
        s.external_event_sources.fetch_add(1, std::memory_order_seq_cst);
      Systematic::cout() << "Add external event source (now "
                         << (prev_count + 1) << ")" << Systematic::endl;
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
      auto prev_count =
        s.external_event_sources.fetch_sub(1, std::memory_order_seq_cst);
      assert(prev_count != 0);
      Systematic::cout() << "Remove external event source (now "
                         << (prev_count - 1) << ")" << Systematic::endl;
    }

    static void set_fair(bool fair)
    {
      Systematic::cout() << "Set fair: " << fair << Systematic::endl;
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

    static T* round_robin()
    {
      static thread_local size_t incarnation;
      static thread_local T* nonlocal;

      if (incarnation != get().incarnation)
      {
        incarnation = get().incarnation;
        nonlocal = get().first_thread;
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
        Systematic::cout() << "Alloc cown during pre-scan" << Systematic::endl;
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
      Systematic::cout() << "Init runtime" << Systematic::endl;

      if ((thread_count != 0) || (count == 0))
        abort();

      thread_count = count;

      // Build a circular linked list of scheduler threads.
      first_thread = new T;
      T* t = first_thread;
      teardown_in_progress = false;

      while (true)
      {
        t->systematic_id = count;
#ifdef USE_SYSTEMATIC_TESTING
        t->systematic_speed_mask =
          (8ULL << (Systematic::get_prng_next() % 4)) - 1;
#endif
        if (count > 1)
        {
          t->next = new T;
          t = t->next;
          count--;
        }
        else
        {
          t->next = first_thread;

          Systematic::cout() << "Runtime initialised" << Systematic::endl;
          break;
        }
      }

      init_barrier();
      sync.init(first_thread);
    }

    void run()
    {
      run_with_startup<>(&nop);
    }

    template<typename... Args>
    void run_with_startup(void (*startup)(Args...), Args... args)
    {
      active_thread_count = thread_count;
      T* t = first_thread;
      {
        ThreadPoolBuilder builder(thread_count);

        Systematic::cout() << "Starting all threads" << Systematic::endl;
        do
        {
          builder.add_thread(&T::run, t, startup, args...);
          t = t->next;
        } while (t != first_thread);

        Systematic::cout() << "All threads started" << Systematic::endl;
      }
      Systematic::cout() << "All threads stopped" << Systematic::endl;

      assert(t == first_thread);
      do
      {
        T* next = t->next;
        delete t;
        t = next;
      } while (t != first_thread);
      Systematic::cout() << "All threads deallocated" << Systematic::endl;

      first_thread = nullptr;
      incarnation++;
#ifdef USE_SYSTEMATIC_TESTING
      Object::reset_ids();
#endif
      thread_count = 0;
      active_thread_count = 0;
      sync.reset();
      state.reset<ThreadState::NotInLD>();

      Epoch::flush(ThreadAlloc::get());
    }

    static bool debug_not_running()
    {
      return get().active_thread_count == 0;
    }

  private:
    inline ThreadState::State next_state(ThreadState::State s)
    {
      return state.next(s, thread_count);
    }

    bool check_for_work()
    {
      // TODO: check for pending async IO
      T* t = first_thread;
      do
      {
        Systematic::cout() << "Checking for pending work on thread "
                           << t->systematic_id << Systematic::endl;
        if (!t->q.is_empty())
        {
          Systematic::cout() << "Found pending work!" << Systematic::endl;
          return true;
        }
        t = t->next;
      } while (t != first_thread);

      Systematic::cout() << "No pending work!" << Systematic::endl;
      return false;
    }

    bool pause(uint64_t tsc)
    {
#ifndef USE_SYSTEMATIC_TESTING
      if ((tsc - last_unpause_tsc) < TSC_PAUSE_SLOP)
        return false;
#else
      UNUSED(tsc);
#endif

      {
        auto h = sync.handle(local());
        if (active_thread_count > 1)
        {
          active_thread_count--;
          Systematic::cout() << "Pausing" << Systematic::endl;
          h.pause();
          Systematic::cout() << "Unpausing" << Systematic::endl;
          active_thread_count++;
          return true;
        }
        Systematic::cout() << "Tried to pause last thread" << Systematic::endl;

        bool has_external_sources =
          external_event_sources.load(std::memory_order_seq_cst) != 0;
        if (has_external_sources)
        {
          assert((runtime_pausing & 1) == 0);
          runtime_pausing++;
          // Ensure this is visible to `unpause` before we check for
          // new work.
          Barrier::memory();
        }

        if (check_for_work())
        {
          // Something has been scheduled LIFO, and the unpause was missed,
          // restart everybody.
          Systematic::cout()
            << "Still work left, back out pause." << Systematic::endl;

          if (has_external_sources)
          {
            assert((runtime_pausing & 1) == 1);
            // Cancel pausing state
            runtime_pausing++;
          }
          h.unpause_all();
          return true;
        }

        if (has_external_sources)
        {
          Systematic::cout() << "Runtime pausing" << Systematic::endl;
          // Wait for external wake-up
          h.pause();

          Systematic::cout() << "Runtime unpausing" << Systematic::endl;
          assert((runtime_pausing & 1) == 1);
          runtime_pausing++;
          return true;
        }

        Systematic::cout() << "Teardown beginning" << Systematic::endl;
        // Used to handle deallocating all the state of the threads.
        teardown_in_progress = true;

        // Tell all threads to stop looking for work.
        T* t = first_thread;
        do
        {
          t->stop();
          t = t->next;
        } while (t != first_thread);
        Systematic::cout() << "Teardown: all threads stopped"
                           << Systematic::endl;

        h.unpause_all();
        Systematic::cout() << "cv_notify_all() for teardown"
                           << Systematic::endl;
      }
      Systematic::cout() << "Teardown: all threads beginning teardown"
                         << Systematic::endl;
      return true;
    }

    bool unpause()
    {
      Systematic::cout() << "unpause()" << Systematic::endl;
      // Work should be added before checking for the runtime_pause.
      Barrier::compiler();

      uint32_t pausing = runtime_pausing;
      if ((pausing & 1) != 0)
      {
        // Prevent starvation by detecting if the pausing state has changed,
        // even if it has paused again.
        do
        {
          sync.handle(local()).unpause_all();
        } while (runtime_pausing == pausing);
        Systematic::cout() << "Unpausing other threads." << Systematic::endl;

        return true;
      }

#ifndef USE_SYSTEMATIC_TESTING
      uint64_t now = Aal::tick();
      uint64_t elapsed = now - last_unpause_tsc;
      last_unpause_tsc = now;

      if (elapsed < TSC_UNPAUSE_SLOP)
        return false;
#endif

      {
        auto h = sync.handle(local());
        if (active_thread_count != thread_count)
        {
          Systematic::cout() << "Unpausing other threads." << Systematic::endl;
          h.unpause_all();
          return true;
        }
        return false;
      }
    }

    void init_barrier()
    {
      barrier_count = thread_count;
    }

    void enter_barrier()
    {
      auto inc = barrier_incarnation;
      {
        auto h = sync.handle(local());
        barrier_count--;
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
