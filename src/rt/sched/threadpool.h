// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "cpu.h"
#include "threadstate.h"

#include <condition_variable>
#include <mutex>
#include <snmalloc.h>

#ifdef USE_SYSTEMATIC_TESTING
#  include "ds/scramble.h"
#  include "test/xoroshiro.h"
#endif

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
    xoroshiro::p128r32 r;
    Scramble scrambler;
#endif

    bool allow_teardown = true;
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
                         << get().inflight_count + 1 << std::endl;
      local()->scheduled_unscanned_cown = true;
      get().inflight_count++;
    }

    static void recv_inflight_message()
    {
      Systematic::cout() << "Decrease inflight count: "
                         << get().inflight_count - 1 << std::endl;
      get().inflight_count--;
    }

    static bool no_inflight_messages()
    {
      Systematic::cout() << "Check inflight count: " << get().inflight_count
                         << std::endl;
      return get().inflight_count == 0;
    }

    static void set_allow_teardown(bool allow)
    {
      Systematic::cout() << "Set allow teardown: " << allow << std::endl;
      auto& s = get();
      s.allow_teardown = allow;
      if (allow)
        s.unpause();
    }

    static void set_fair(bool fair)
    {
      Systematic::cout() << "Set fair: " << fair << std::endl;
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
    static size_t rand_get_next()
    {
      return get().r.next();
    }

    static Scramble& get_scrambler()
    {
      return (get().scrambler);
    }

    /// 1/2^range_bits likelyhood of coin saying true
    static bool coin(size_t range_bits = 1)
    {
      assert(range_bits < 20);
      return (rand_get_next() & ((1ULL << range_bits) - 1)) == 0;
    }

    void set_seed(uint64_t seed)
    {
      r.set_state(seed);
      scrambler.setup(r);
    }

    void choose_thread()
    {
      std::unique_lock<std::mutex> lock(m);
      assert(running_thread != nullptr);
      uint32_t i = r.next() % thread_count;
      auto result = running_thread;

      while (i > 0 || result->sleeping)
      {
        result = result->next;
        if (i != 0)
          i--;
        if (result == running_thread)
          continue;
        if (result->sleeping)
          continue;
      }
      running_thread = result;
      assert(!(running_thread->sleeping));
      result->cv.notify_all();
    }

    void yield_my_turn_inner()
    {
      auto me = local();
      if (me == nullptr)
        return;

      assert(running_thread == me || running_thread == nullptr);

      uint32_t next = r.next() & me->systematic_speed_mask;
      if (next == 0 && running_thread != nullptr)
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
      while (running_thread != nullptr && running_thread != me)
        me->cv.wait(lock);
    }

    static void wait_for_my_first_turn()
    {
      auto me = local();
      get().enter_barrier();
      assert(me != nullptr);
      get().wait_for_my_turn_inner(me);
    }

    /// Used to simulate waiting on the thread pools condition variable for more
    /// work.
    static void cv_wait()
    {
      auto& sched = get();
      auto me = local();
      assert(!(me->sleeping));
      me->sleeping = true;

      assert(sched.running_thread == me);

      auto curr = me;
      while (curr->sleeping)
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
        curr->sleeping = false;
        curr = curr->next;
      } while (curr != head);

      // Can be signalled from outside the runtime if external work is injected
      // if this is a runtime thread, then yield.
      if (local() != nullptr)
        yield_my_turn();
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
        Systematic::cout() << "Alloc cown during pre-scan" << std::endl;
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

      while (count > 1)
      {
        t->next = new T;
        t->systematic_id = count;
        t = t->next;
        count--;
      }
      t->systematic_id = count;
#ifdef USE_SYSTEMATIC_TESTING
      t->systematic_speed_mask = (1 << (rand_get_next() % 16)) - 1;
#endif
      t->next = first_thread;
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

      Systematic::cout() << "Starting all threads" << std::endl;

      do
      {
        t->template start<Args...>(topology.get(i++), startup, args...);
        t = t->next;
      } while (t != first_thread);

      t = first_thread;

      do
      {
        T* next = t->next;
        delete t;
        t = next;
      } while (t != first_thread);
      Systematic::cout() << "All threads stopped" << std::endl;

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
        Systematic::cout() << "Pausing" << std::endl;
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
          Systematic::cout() << "Unpausing" << std::endl;
          return true;
        }

        // TODO: check for pending async IO
        T* t = first_thread;
        do
        {
          if (!t->q.is_empty())
          {
// Something has been scheduled LIFO, and the unpause was missed,
// restart everybody.
#ifdef USE_SYSTEMATIC_TESTING
            lock.unlock();
            cv_notify_all();
#else
            cv.notify_all();
#endif
            return true;
          }
          t = t->next;
        } while (t != first_thread);

        if (!allow_teardown)
        {
          assert((runtime_pausing & 1) == 0);
          runtime_pausing++;
          Barrier::memory();

          t = first_thread;
          do
          {
            if (!t->q.is_empty())
            {
              Systematic::cout() << "Still work left" << std::endl;
              runtime_pausing++;
#ifdef USE_SYSTEMATIC_TESTING
              cv_notify_all();
#else
              cv.notify_all();
#endif
              return true;
            }
            t = t->next;
          } while (t != first_thread);

          Systematic::cout() << "Runtime pausing" << std::endl;
          cv.wait(lock);

          Systematic::cout() << "Runtime unpausing" << std::endl;
          runtime_pausing++;
          cv.notify_all();

          return true;
        }

        // Used to handle deallocating all the state of the threads.
        Systematic::cout() << "Teardown beginning" << std::endl;
        teardown_in_progress = true;

        t = first_thread;
#ifdef USE_SYSTEMATIC_TESTING
        running_thread = nullptr;
#endif
        do
        {
          t->stop();
          t = t->next;
        } while (t != first_thread);
        Systematic::cout() << "Teardown: all threads stopped" << std::endl;
      }
      Systematic::cout() << "cv_notify_all() for teardown" << std::endl;
#ifdef USE_SYSTEMATIC_TESTING
      T* t = first_thread;
      do
      {
        t->cv.notify_all();
        t = t->next;
      } while (t != first_thread);
#else
      cv.notify_all();
#endif
      Systematic::cout() << "Teardown: all threads beginning teardown"
                         << std::endl;
      return true;
    }

    bool unpause()
    {
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
        Systematic::cout() << "Unpausing other threads." << std::endl;

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
      Systematic::cout() << "Unpausing other threads." << std::endl;

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
          cv.wait(lock);
          return;
        }
        barrier_count = thread_count;
      }
      cv.notify_all();
    }
  };
} // namespace verona::rt
