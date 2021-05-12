// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "../pal/threadpoolbuilder.h"
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
    Finished
  };

  inline snmalloc::function_ref<bool()> true_thunk{[]() { return true; }};

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
    uint64_t barrier_incarnation = 0;

    T* first_thread = nullptr;
#ifdef USE_SYSTEMATIC_TESTING
    /// Specifies which thread is currently executing in systematic testing
    /// nullptr is used to mean no thread is currently running.
    T* running_thread = nullptr;
    /// Used to prevent systematic testing attempt to access threads when the
    /// runtime has been deallocated.
    bool shutdown = true;
    /// Mutex for manipulating systematic testing datastructures
    ///  * running_thread
    ///  * systematic_status
    ///  * shutdown
    std::mutex m_sys;

    /// Notify incarnation
    /// Complete wrap around will lead to lost wake-up.  This seems safe to
    /// ignore.
    std::atomic<uint64_t> cv_incarnation = 0;
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
    /// A message can be enqueued before the runtime is running if there is a external
    /// event source from the start.
    static void add_external_event_source()
    {
      auto& s = get();
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

#ifdef USE_SYSTEMATIC_TESTING
    /// Must hold the systematic testing lock to call this.
    /// External is used for calls from outside a scheduler thread.
    void choose_thread(std::unique_lock<std::mutex>&, bool external = false)
    {
      if (external && get().running_thread != nullptr)
      {
        // Runtime was not asleep, so don't try to wake up a thread.
        return;
      }

      uint32_t i = Systematic::get_prng_next() % thread_count;
      auto start = running_thread;
      assert(local() == start);
      if (start == nullptr)
      {
        // This is an external wake up
        start = first_thread;
      }

      // Skip to a first choice for selecting.
      while (i > 0)
      {
        start = start->next;
        i--;
      }

      auto result = start;
      while ((result->systematic_state != SystematicState::Active) ||
             !result->guard())
      {
        result = result->next;
        if (result == start)
        {
          // If all threads are waiting, then there must be an external event
          // source.  Or we are in a set up phase and the runtime is not running
          // any threads yet.
          assert((external_event_sources > 0) || (running_thread == nullptr));
          running_thread = nullptr;
          Systematic::cout() << "All threads sleeping!" << Systematic::endl;
          return;
        }
      }
      Systematic::cout() << "Set running thread:" << result->systematic_id
                         << Systematic::endl;
      assert(result->guard());

      running_thread = result;
      assert(result->systematic_state == SystematicState::Active);
      result->cv.notify_all();
    }

    /// lock, must be holding the m_sys mutex.
    /// Will only pass control back to this thread once the guard g has been
    /// established.
    void yield_until(
      T*& me,
      std::unique_lock<std::mutex>& lock,
      snmalloc::function_ref<bool()> g)
    {
      assert(lock.mutex() == &m_sys);
      me->guard = g;
      choose_thread(lock);
      wait_for_my_turn_inner(lock, me);
      me->guard = true_thunk;
    }

    static void yield_my_turn()
    {
      auto me = local();
      if (me == nullptr)
        return;

      assert(get().running_thread == me);

      if (me->steps == 0)
      {
        auto& sched = get();
        std::unique_lock<std::mutex> lock(sched.m_sys);
        sched.yield_until(me, lock, true_thunk);
        me->steps = Systematic::get_prng_next() & me->systematic_speed_mask;
      }
      else
      {
        me->steps--;
      }
    }

    void wait_for_my_turn_inner(std::unique_lock<std::mutex>& lock, T* me)
    {
      assert(lock.mutex() == &m_sys);
      Systematic::cout() << "Waiting for turn" << Systematic::endl;
      while (running_thread != me)
        me->cv.wait(lock);
      Systematic::cout() << "Now my turn" << Systematic::endl;
      assert(me->systematic_state == SystematicState::Active);
    }

    static void wait_for_my_first_turn()
    {
      auto me = local();
      assert(me != nullptr);
      std::unique_lock<std::mutex> lock(get().m_sys);
      get().wait_for_my_turn_inner(lock, me);
    }

    /// Used to simulate waiting on the thread pools condition variable for more
    /// work.
    static void cv_wait()
    {
      auto& sched = get();
      auto me = local();

      Systematic::cout() << "Waiting state" << Systematic::endl;
      assert(me->systematic_state == SystematicState::Active);
      {
        std::unique_lock<std::mutex> lock(sched.m_sys);
        auto incarnation = sched.cv_incarnation.load();
        auto guard = [incarnation]() {
          return incarnation != get().cv_incarnation.load();
        };
        // Guard should not hold here.
        assert(!guard());
        sched.yield_until(me, lock, guard);
      }
      Systematic::cout() << "Notified" << Systematic::endl;

      assert(sched.running_thread == me);
    }

    /// Used to simulate waking all waiting threads on the thread pools
    /// condition variable.
    static void cv_notify_all()
    {
      auto& sched = get();
      if (sched.shutdown)
        return;

      // Treat as a yield pointer if thread is under systematic testing control.
      if (local() != nullptr)
      {
        Systematic::cout() << "cv_notify_all internal" << Systematic::endl;
        sched.cv_incarnation++;
        yield_my_turn();
      }
      else
      {
        Systematic::cout() << "cv_notify_all external" << Systematic::endl;
        // Can be signalled from outside the runtime if external work is
        // injected if this is a runtime thread, then yield.
        // This will wake a thread if none are currently running, otherwise
        // does nothing.
        // m_sys mutex is required to prevent lost wake-up
        std::unique_lock<std::mutex> lock(sched.m_sys);
        sched.cv_incarnation++;
        sched.choose_thread(lock, true);
      }
    }

    static void thread_finished()
    {
      auto& sched = get();
      auto me = local();
      std::unique_lock<std::mutex> lock(sched.m_sys);
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
          sched.shutdown = true;
          return;
        }
      }
      sched.choose_thread(lock);
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
      Systematic::cout() << "Init runtime" << Systematic::endl;

      if ((thread_count != 0) || (count == 0))
        abort();

      // Build a circular linked list of scheduler threads.
      first_thread = new T;
      T* t = first_thread;
      teardown_in_progress = false;

#ifdef USE_SYSTEMATIC_TESTING
      running_thread = nullptr;
#endif
      size_t i = count;
      while (true)
      {
        t->systematic_id = i;
#ifdef USE_SYSTEMATIC_TESTING
        t->systematic_speed_mask =
          (8ULL << (Systematic::get_prng_next() % 4)) - 1;
#endif
        if (i > 1)
        {
          t->next = new T;
          t = t->next;
          i--;
        }
        else
        {
          t->next = first_thread;

          Systematic::cout() << "Runtime initialised" << Systematic::endl;
          break;
        }
      }

      thread_count = count;
      active_thread_count = thread_count;

      init_barrier();
#ifdef USE_SYSTEMATIC_TESTING
      {
        std::unique_lock<std::mutex> lock(m_sys);
        shutdown = false;
        choose_thread(lock, true);
      }
#endif
    }

    void run()
    {
      run_with_startup<>(&nop);
    }

    template<typename... Args>
    void run_with_startup(void (*startup)(Args...), Args... args)
    {
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
#ifdef USE_SYSTEMATIC_TESTING
          lock.unlock();
          cv_wait();
          lock.lock();
#else
          cv.wait(lock);
#endif

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
      auto inc = barrier_incarnation;
      {
        std::unique_lock<std::mutex> lock(m);
        barrier_count--;
        if (barrier_count != 0)
        {
          while (inc == barrier_incarnation)
          {
#ifdef USE_SYSTEMATIC_TESTING
            lock.unlock();
            cv_wait();
            lock.lock();
#else
            cv.wait(lock);
#endif
          }
          return;
        }
        barrier_count = thread_count;
      }

      barrier_incarnation++;
#ifdef USE_SYSTEMATIC_TESTING
      cv_notify_all();
#else
      cv.notify_all();
#endif
    }
  };
} // namespace verona::rt
