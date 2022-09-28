// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "../pal/semaphore.h"
#include "ds/scramble.h"
#include "test/xoroshiro.h"

namespace verona::rt
{
  class Systematic
  {
    enum class SystematicState
    {
      Active,
      Finished
    };

#ifdef USE_SYSTEMATIC_TESTING
    static constexpr bool enabled = true;
#else
    static constexpr bool enabled = false;
#endif

  public:
    /**
     * Per thread state required for systematic testing.
     */
    class Local
    {
      friend Systematic;
      /// Used by systematic testing to implement the condition variable,
      /// and thread termination.
      SystematicState systematic_state = SystematicState::Active;

      /// Used to specify a condition when this thread should/could make
      /// progress.  It is used to implement condition variables.
      snmalloc::function_ref<bool()> guard = true_thunk;

      /// How many uninterrupted steps this threads has been selected to run
      /// for.
      size_t steps = 0;

      /// Alters distribution of steps taken in systematic testing.
      size_t systematic_speed_mask = 1;

      /// Used for debugging.
      size_t systematic_id;

      /// Disables yielding.
      bool no_yield = false;

      /// Used to sleep and wake the threads systematically.
      pal::SleepHandle sh;

      // Pointer to cyclic list of threads structures.
      Local* next = nullptr;

      Local(size_t id) : systematic_id(id) {}
    };

    static inline snmalloc::function_ref<bool()> true_thunk{
      []() { return true; }};

  private:
    /// Currently running thread.  Points to a cyclic list of all the threads.
    static inline Local* running_thread{nullptr};

    /// Contains thread local state primarily for sleeping and waking a thread.
    static inline thread_local Local* local_systematic{nullptr};

    /// How many threads have been added to this systematic testing run.
    static inline size_t num_threads{0};

    /// How many threads have terminated in this test run.
    static inline size_t finished_threads{0};

    /// If true, then systematic testing has enabled a thread.
    static inline bool running{false};

    /// Return a mutable reference to the pseudo random number generator (PRNG).
    /// It is assumed that the PRNG will only be setup once via `set_seed`.
    /// After it is setup, the PRNG must only be used via `get_prng_next`.
    static xoroshiro::p128r32& get_prng_for_setup()
    {
      static xoroshiro::p128r32 prng;
      return prng;
    }

    /// Return a mutable reference to the scrambler. It is assumed that the
    /// scrambler will only be setup once via `set_seed`. After it is setup, the
    /// scrambler must only be accessed via a const reference
    /// (see `get_scrambler`).
    static verona::Scramble& get_scrambler_for_setup()
    {
      static verona::Scramble scrambler;
      return scrambler;
    }

  public:
    /// Return the next pseudo random number.
    static uint32_t get_prng_next()
    {
      auto& prng = get_prng_for_setup();
      static snmalloc::FlagWord lock;
      snmalloc::FlagLock l{lock};
      return prng.next();
    }

    /// Return a const reference to the scrambler.
    static const verona::Scramble& get_scrambler()
    {
      return get_scrambler_for_setup();
    }

    static void set_seed(uint64_t seed)
    {
      auto& rng = get_prng_for_setup();
      rng.set_state(seed);
      get_scrambler_for_setup().setup(rng);
    }

    /// 1/(2^range_bits) likelyhood of returning true
    static bool coin(size_t range_bits = 1)
    {
      assert(range_bits < 20);
      return (get_prng_next() & ((1ULL << range_bits) - 1)) == 0;
    }

  private:
    /**
     * Internal function for selecting the next thread to run.
     *
     * startup is true if this is part of starting the sys-testing
     * runtime.
     */
    static void choose_next(bool startup = false)
    {
      auto r = get_prng_next();
      auto i = snmalloc::bits::ctz(r != 0 ? r : 1);
      auto start = running_thread;

      assert((running_thread == local_systematic) || startup);
      snmalloc::UNUSED(startup);

      // Skip to a first choice for selecting.
      for (; i > 0; i--)
        start = start->next;

      auto curr = start;
      while ((curr->systematic_state != SystematicState::Active) ||
             !curr->guard())
      {
        curr = curr->next;
        if (curr == start)
        {
          Logging::cout() << "All threads sleeping!" << Logging::endl;
          abort();
        }
      }

      Logging::cout() << "Set running thread:" << curr->systematic_id
                      << Logging::endl;
      assert(curr->guard());

      running_thread = curr;
      assert(curr->systematic_state == SystematicState::Active);
      curr->steps = get_prng_next() & curr->systematic_speed_mask;
      curr->sh.wake();
    }

  public:
    /**
     * Creates the structure for pausing a thread in systematic testing.  This
     * should be called in a sequential setting so that determinism is
     * maintained.
     */
    static Local* create_systematic_thread(size_t id)
    {
      if constexpr (enabled)
      {
        assert((!running) || (running_thread == local_systematic));

        auto l = new Local(id);
        if (running_thread == nullptr)
        {
          // First thread. Create single element cyclic list.
          l->next = l;
          running_thread = l;
        }
        else
        {
          // Insert into the cyclic list.
          l->next = running_thread->next;
          running_thread->next = l;
        }
        l->systematic_speed_mask = (8ULL << (get_prng_next() % 4)) - 1;

        num_threads++;
        return l;
      }
      else
      {
        snmalloc::UNUSED(id);
        return nullptr;
      }
    }

    /**
     * Attach this thread to the systematic testing implementation.
     *
     * Must be passed a pointer return from `create_systematic_thread`.
     */
    static void attach_systematic_thread(Local* l)
    {
      if constexpr (enabled)
      {
        local_systematic = l;
        l->sh.sleep();
      }
      else
      {
        snmalloc::UNUSED(l);
      }
    }

    /**
     * Switches thread in systematic testing and only returns once
     * guard evaluates to true.
     */
    static void yield_until(snmalloc::function_ref<bool()> guard)
    {
      if constexpr (enabled)
      {
        if (!running)
        {
          if (!guard())
            abort();
          return;
        }

        assert(local_systematic != nullptr);

        if (
          (local_systematic->no_yield || local_systematic->steps > 0) &&
          guard())
        {
          if (!local_systematic->no_yield)
            local_systematic->steps--;
          return;
        }

        local_systematic->guard = guard;

        choose_next();

        local_systematic->sh.sleep();
      }
      else
      {
        snmalloc::UNUSED(guard);
      }
    }

    /**
     * Switches thread in systematic testing.
     */
    static void yield()
    {
      yield_until(true_thunk);
    }

    /**
     * Call this when the thread has completed.
     */
    static void finished_thread()
    {
      if constexpr (enabled)
      {
        finished_threads++;

        if (finished_threads < num_threads)
        {
          local_systematic->systematic_state = SystematicState::Finished;
          choose_next();
          local_systematic->sh.sleep();
        }
        else
        {
          // All threads have finished.
          auto curr = running_thread;
          running = false;
          Logging::cout() << "All threads finished!" << Logging::endl;
          do
          {
            auto next = curr->next;
            Logging::cout() << "Thread " << curr->systematic_id << " finished."
                            << Logging::endl;
            curr->sh.wake();
            curr = next;
          } while (curr != running_thread);

          running_thread = nullptr;
          num_threads = 0;
          finished_threads = 0;
        }

        delete local_systematic;
        local_systematic = nullptr;
      }
    }

    /**
     * Stops the step counter from decreasing
     *
     * Used by tests to force a series of events to happen without interleaving.
     */
    static void disable_yield()
    {
      if constexpr (enabled)
      {
        assert(local_systematic->no_yield == false);
        local_systematic->no_yield = true;
      }
    }

    /**
     * Restarts the step counter.
     * Should be called after disable_yield to re-enable interleaving.
     */
    static void enable_yield()
    {
      if constexpr (enabled)
      {
        assert(local_systematic->no_yield == true);
        local_systematic->no_yield = false;
      }
    }

    /**
     * Call this to start threads running in systematic testing.
     */
    static void start()
    {
      if constexpr (enabled)
      {
        running = true;
        choose_next(true);
      }
    }
  };
} // namespace verona::rt
