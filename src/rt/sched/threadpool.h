// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "cpu.h"
#include "test/systematic.h"
#include "threadstate.h"

#include <condition_variable>
#include <mutex>
#include <snmalloc.h>

namespace verona::rt
{
  enum class SystematicState
  {
    Active,
    Waiting,
    Finished
  };

  /// Used for default prerun for a thread.
  inline void nop() {}

  using namespace snmalloc;
  template<class T>
  class ThreadPool
  {
  private:
    friend T;

    static constexpr uint64_t TSC_PAUSE_SLOP = 1'000'000;
    static constexpr uint64_t TSC_UNPAUSE_SLOP = TSC_PAUSE_SLOP / 2;

    bool detect_leaks = true;
    size_t incarnation = 1;
    size_t thread_count = 0;
    size_t active_thread_count = 0;

    /**
     * Number of messages that have been sent that may not be visible to a
     *thread in a Scan state.
     **/
    std::atomic<size_t> inflight_count = 0;

    uint64_t last_unpause_tsc = Aal::tick();
    std::mutex m;
    std::condition_variable cv;
    std::atomic_uint64_t barrier_count = 0;
    T* first_thread = nullptr;
#ifdef USE_SYSTEMATIC_TESTING
    T* running_thread = nullptr;
#endif

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
    Topology topology;

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
    /// This should only be called before the runtime has started, or when the
    /// caller can guarantee there is at least one other external_event_source
    /// for the duration of this call.
    static void add_external_event_source()
    {
      auto& s = get();
      assert((s.external_event_sources != 0) || (s.debug_not_running()));
      auto prev_count =
        s.external_event_sources.fetch_add(1, std::memory_order_seq_cst);
      Systematic::cout() << "Add external event source (now "
                         << (prev_count + 1) << ")" << Systematic::endl;
    }

    /// Decrement the external event source count. This will allow runtime
    /// teardown if the count drops to zero.
    static void remove_external_event_source()
    {
      auto& s = get();
      auto prev_count =
        s.external_event_sources.fetch_sub(1, std::memory_order_seq_cst);
      assert(prev_count != 0);
      Systematic::cout() << "Remove external event source (now "
                         << (prev_count - 1) << ")" << Systematic::endl;
      if (prev_count == 1)
        s.unpause();
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

#ifdef USE_SYSTEMATIC_TESTING
    void choose_thread()
    {
      std::unique_lock<std::mutex> lock(m);
      assert(running_thread != nullptr);
      uint32_t i = Systematic::get_prng_next() % thread_count;
      auto result = running_thread;

      while (i > 0 || (result->systematic_state != SystematicState::Active))
      {
        result = result->next;
        if (i != 0)
          i--;
        if (result == running_thread)
          continue;
        if (result->systematic_state != SystematicState::Active)
          continue;
      }
      running_thread = result;
      assert(result->systematic_state == SystematicState::Active);
      result->cv.notify_all();
    }

    void yield_my_turn_inner()
    {
      auto me = local();
      if (me == nullptr)
        return;

      assert(running_thread == me);

      uint32_t next = Systematic::get_prng_next() & me->systematic_speed_mask;
      if (next == 0)
      {
        choose_thread();
        wait_for_my_turn_inner(me);
      }
    }

    static void yield_my_turn()
    {
      get().yield_my_turn_inner();
    }

    void wait_for_my_turn_inner(T* me)
    {
      std::unique_lock<std::mutex> lock(m);
      while (running_thread != me)
        me->cv.wait(lock);
    }

    static void wait_for_my_first_turn()
    {
      auto me = local();
      assert(me != nullptr);
      get().wait_for_my_turn_inner(me);
    }

    /// Used to simulate waiting on the thread pools condition variable for more
    /// work.
    static void cv_wait()
    {
      auto& sched = get();
      auto me = local();
      assert(me->systematic_state != SystematicState::Waiting);
      me->systematic_state = SystematicState::Waiting;

      assert(sched.running_thread == me);

      // Confirm at least one other thread is running,
      // otherwise we have deadlocked the system.
      // Tests for external IO might fail on this check, and something might
      // need adding.
      auto curr = me;
      while (curr->systematic_state != SystematicState::Active)
      {
        curr = curr->next;
        assert(curr != me);
      }

      sched.choose_thread();
      sched.wait_for_my_turn_inner(me);
    }

    /// Used to simulate waking all waiting threads on the thread pools
    /// condition variable.
    static void cv_notify_all()
    {
      auto head = get().first_thread;
      auto curr = head;
      do
      {
        if (curr->systematic_state == SystematicState::Waiting)
          curr->systematic_state = SystematicState::Active;
        curr = curr->next;
      } while (curr != head);

      // Can be signalled from outside the runtime if external work is injected
      // if this is a runtime thread, then yield.
      if (local() != nullptr)
        yield_my_turn();
    }

    static void thread_finished()
    {
      auto& sched = get();
      auto me = local();
      assert(me->systematic_state == SystematicState::Active);
      me->systematic_state = SystematicState::Finished;

      assert(sched.running_thread == me);

      Systematic::cout() << "Thread finished." << Systematic::endl;

      // Confirm at least one other thread is running,
      auto curr = me;
      while (curr->systematic_state != SystematicState::Active)
      {
        curr = curr->next;
        if (curr == me)
        {
          Systematic::cout() << "Last thread finished." << Systematic::endl;
          // No threads left
          sched.running_thread = nullptr;
          return;
        }
      }

      sched.choose_thread();
    }
#endif

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
      if ((thread_count != 0) || (count == 0))
        abort();

      // Build a circular linked list of scheduler threads.
      thread_count = count;
      first_thread = new T;
      T* t = first_thread;
      teardown_in_progress = false;

#ifdef USE_SYSTEMATIC_TESTING
      running_thread = first_thread;
#endif

      while (true)
      {
        t->systematic_id = count;
#ifdef USE_SYSTEMATIC_TESTING
        t->systematic_speed_mask =
          (1 << (Systematic::get_prng_next() % 16)) - 1;
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
          return;
        }
      }
    }

    void run()
    {
      run_with_startup<>(&nop);
    }

    template<typename... Args>
    void run_with_startup(void (*startup)(Args...), Args... args)
    {
      topology.acquire();
      active_thread_count = thread_count;

      init_barrier();
#ifdef USE_SYSTEMATIC_TESTING
      choose_thread();
#endif

      size_t i = 0;
      T* t = first_thread;

      Systematic::cout() << "Starting all threads" << Systematic::endl;
      do
      {
        t->template start<Args...>(topology.get(i++), startup, args...);
        t = t->next;
      } while (t != first_thread);
      Systematic::cout() << "All threads started" << Systematic::endl;

      assert(t == first_thread);
      do
      {
        t->block_until_finished();
        t = t->next;
      } while (t != first_thread);
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
      state.reset<ThreadState::NotInLD>();
      topology.release();

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
        if (!t->q.is_empty())
        {
          return true;
        }
        t = t->next;
      } while (t != first_thread);
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
        std::unique_lock<std::mutex> lock(m);
        Systematic::cout() << "Pausing" << Systematic::endl;
        if (active_thread_count > 1)
        {
          active_thread_count--;
#ifdef USE_SYSTEMATIC_TESTING
          lock.unlock();
          cv_wait();
          lock.lock();
#else
          cv.wait(lock);
#endif
          active_thread_count++;
          Systematic::cout() << "Unpausing" << Systematic::endl;
          return true;
        }

        bool has_external_sources = false;
        if (external_event_sources.load(std::memory_order_seq_cst) != 0)
        {
          assert((runtime_pausing & 1) == 0);
          runtime_pausing++;
          // Ensure this is visible to `unpause` before we check for
          // new work.
          Barrier::memory();
          has_external_sources = true;
        }

        if (check_for_work())
        {
          // Something has been scheduled LIFO, and the unpause was missed,
          // restart everybody.
          Systematic::cout()
            << "Still work left, back out pause." << Systematic::endl;

          if (has_external_sources)
          {
            // Cancel pausing state
            runtime_pausing++;
          }
#ifdef USE_SYSTEMATIC_TESTING
          lock.unlock();
          cv_notify_all();
#else
          cv.notify_all();
#endif
          return true;
        }

        if (has_external_sources)
        {
          Systematic::cout() << "Runtime pausing" << Systematic::endl;
          // Wait for external wake-up
          cv.wait(lock);

          Systematic::cout() << "Runtime unpausing" << Systematic::endl;
          runtime_pausing++;
          cv.notify_all();

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
      }
      Systematic::cout() << "cv_notify_all() for teardown" << Systematic::endl;
#ifdef USE_SYSTEMATIC_TESTING
      cv_notify_all();
#else
      cv.notify_all();
#endif
      Systematic::cout() << "Teardown: all threads beginning teardown"
                         << Systematic::endl;
      return true;
    }

    bool unpause()
    {
      // Work should be added before checking for the runtime_pause.
      Barrier::compiler();

      uint32_t pausing = runtime_pausing;
      if ((pausing & 1) != 0)
      {
        // Prevent starvation by detecting if the pausing state has changed,
        // even if it has paused again.
        do
        {
#ifdef USE_SYSTEMATIC_TESTING
          cv_notify_all();
#else
          cv.notify_all();
#endif
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
        std::unique_lock<std::mutex> lock(m);

        if (active_thread_count == thread_count)
          return false;
      }

#ifdef USE_SYSTEMATIC_TESTING
      cv_notify_all();
#else
      cv.notify_all();
#endif
      Systematic::cout() << "Unpausing other threads." << Systematic::endl;

      return true;
    }

    void init_barrier()
    {
      barrier_count = thread_count;
    }

    void enter_barrier()
    {
      {
        std::unique_lock<std::mutex> lock(m);
        barrier_count--;
        if (barrier_count != 0)
        {
#ifdef USE_SYSTEMATIC_TESTING
          lock.unlock();
          cv_wait();
          lock.lock();
#else
          cv.wait(lock);
#endif
          return;
        }
        barrier_count = thread_count;
      }
#ifdef USE_SYSTEMATIC_TESTING
      cv_notify_all();
#else
      cv.notify_all();
#endif
    }
  };
} // namespace verona::rt
