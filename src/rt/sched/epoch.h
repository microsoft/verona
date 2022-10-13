// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

/**
 * This file provides an Epoch based memory reclamation mechanism.
 *
 * There is a global epoch, and each thread has its own view of the local
 * epoch. The local epoch can be in one of four states
 *   1. Equal to the global epoch
 *   2. One less that the global epoch
 *   3. Ejected
 *   4. Rejoining
 * The global epoch can be advanced as long as there is no thread in state 2.
 * When a thread is in a state 2, any other thread that wishes to advance
 * the epoch must eject that thread. If a thread is using an Epoch it cannot be
 * ejected, but if it is not using an Epoch then it may be ejected.
 *
 * The ejection mechanism is built on top of the Asymetric lock.  Holding the
 * epoch, holds the `internal` side of a lock.  To eject a thread, the ejectee
 * must hold the `external` side of the lock.
 *
 * To rejoin a thread must observe itself to be in state 1. That is set its
 * local epoch to the global one, and then observe the global one has not
 * changed. A rejoining thread may prevent a thread from advancing the epoch.
 */

#include "../ds/asymlock.h"
#include "../ds/queue.h"
#include "../test/logging.h"
#include "region/immutable.h"

#include <snmalloc/snmalloc.h>

namespace verona::rt
{
  /**
   * The representation of an epoch is a 63 bit counter.  If the top bit is
   * set, then the epoch is considered ejected.
   */
  static constexpr uint64_t EJECTED_BIT = 0x8000000000000000;

  static uint64_t inc_epoch_by(uint64_t epoch, uint64_t i)
  {
    // Maybe overflowing after the addition; need ~EJECTED_BIT for modulo
    return (epoch + i) & ~EJECTED_BIT;
  }

  /**
   * Represents the global state of the current epoch.
   */
  class GlobalEpoch
  {
  private:
    friend class LocalEpoch;

    static std::atomic<uint64_t>& global_epoch()
    {
      static std::atomic<uint64_t> global_epoch;
      return global_epoch;
    }

    static void set(uint64_t e)
    {
      Logging::cout() << "Global epoch set to " << e << std::endl;
      global_epoch().store(e, std::memory_order_release);
    }

  public:
    static uint64_t get()
    {
      return global_epoch().load(std::memory_order_acquire);
    }

    static bool is_outdated(uint64_t e)
    {
      auto global_e = get();

      for (uint64_t i = 0; i < 3; ++i)
      {
        if (inc_epoch_by(e, i) == global_e)
        {
          return false;
        }
      }
      return true;
    }
  };

  /**
   * Represents the thread local state of the thread's current epoch.
   *
   * This also stores the delayed operations that need to wait until the epoch
   * has advanced sufficiently.
   *
   * There are two types of delayed operations:
   *   1. Decrementing the reference count of an object
   *   2. Deallocating an object.
   */
  class LocalEpoch : public Pooled<LocalEpoch>
  {
  private:
    /**
     * Used to represent objects that are being delayed deallocations.
     * The object itself is used to represent this. The object to be deallocated
     * is used itself by casting to an InnerNode.
     */
    struct InnerNode
    {
      InnerNode* next;
    };

    /**
     * Represents the node in the queue for objects that need an incref
     * applying.
     */
    struct DecNode
    {
      DecNode* next;
      Object* o;
    };

    friend class ThreadLocalEpoch;
    friend class Epoch;

    /// Ranges from 0 to 3 to represent the current epoch on
    /// more finite structures.
    uint8_t index = 0;

    /// The queue of objects to be deallocated.
    Queue<InnerNode> delete_list;

    /// Represents how many objects in each epoch of delete_list
    size_t unusable[4] = {0, 0, 0, 0};

    /// The queue of objects to be decremeneted.
    Queue<DecNode> dec_list;

    /// Represents how many objects in each epoch of dec_list
    size_t to_dec[4] = {0, 0, 0, 0};

    // Providing heuristic for advancing the epoch. Currently, we only look at
    // one slot to determine if we should advance the epoch (see
    // advance_is_sensible()), but we keep the history here so that better
    // heuristic could be applied later.
    size_t pressure[4] = {0, 0, 0, 0};

    /// The current epoch for this structure.  Initially set to EJECTED_BIT
    /// so we hit a slow path initially.
    std::atomic<uint64_t> epoch = EJECTED_BIT;

    /// Used to allow ejection from other threads.
    AsymmetricLock lock;

    /// Used to check that all threads are in a particular state.
    /// Forward reference due to requiring the LocalEpochPool to
    /// find all LocalEpochs.
    template<typename T, bool predicate(LocalEpoch* p, T t)>
    static bool forall(T t);

    void add_to_delete_list(void* p)
    {
      delete_list.enqueue((InnerNode*)p);
      (*get_unusable(2))++;
      (*get_pressure(2))++;
      debug_check_count();
    }

    void add_to_dec_list(Alloc& alloc, Object* p)
    {
      auto node = (DecNode*)alloc.alloc<sizeof(DecNode)>();
      node->o = p;
      dec_list.enqueue(node);
      (*get_to_dec(2))++;
      debug_check_count();
    }

    inline void use_epoch(Alloc& a)
    {
      if (lock.internal_acquire())
      {
        auto new_epoch = GlobalEpoch::get();
        auto old_epoch = get_epoch();

        if (old_epoch != new_epoch)
          use_epoch_rare(a, old_epoch);
      }
    }

    inline void release_epoch(Alloc& a)
    {
      if ((lock.internal_count() == 1) && (advance_is_sensible()))
        release_epoch_rare(a);

      lock.internal_release();
    }

    size_t* get_pressure(uint8_t i)
    {
      return &pressure[(index + i) & 3];
    }

    size_t* get_unusable(uint8_t i)
    {
      return &unusable[(index + i) & 3];
    }

    size_t* get_to_dec(uint8_t i)
    {
      return &to_dec[(index + i) & 3];
    }

    /**
     * Deals with the old epoch's delayed operations for this thread.
     */
    void flush_old_epoch(Alloc& alloc)
    {
      debug_check_count();

      {
        auto cell = get_unusable(0);
        auto usable = *cell;

        for (size_t n = 0; n < usable; n++)
        {
          auto d = delete_list.dequeue();
          Logging::cout() << "Delayed delete on " << d << Logging::endl;
          alloc.dealloc(d);
        }
        *cell = 0;

        *get_pressure(0) = 0;
      }

      {
        auto cell = get_to_dec(0);
        auto usable = *cell;

        for (size_t n = 0; n < usable; n++)
        {
          auto dn = (DecNode*)dec_list.dequeue();
          // Reestablish invariant.  The Immutable::release below
          // can re-enter the Epoch structure so we need to ensure the
          // invariant is re-established.
          (*cell)--;
          auto o = dn->o;
          alloc.dealloc<sizeof(DecNode)>(dn);
          Logging::cout() << "Delayed decref on " << o << Logging::endl;
          Immutable::release(alloc, o);
        }
      }
      debug_check_count();

      index = (index + 1) & 3;

      debug_check_count();
    }

    // TODO: Add a proper heuristic here
    bool advance_is_sensible()
    {
#ifdef USE_SYSTEMATIC_TESTING
      return Systematic::coin(2);
#else
      return *get_pressure(2) > 128;
#endif
    }

    // TODO: Add a proper heuristic here
    bool advance_is_urgent()
    {
#ifdef USE_SYSTEMATIC_TESTING
      return Systematic::coin(2);
#else
      return *get_pressure(2) > 1024;
#endif
    }

    uint64_t get_epoch()
    {
      return epoch.load(std::memory_order_acquire);
    }

    /**
     * Used to advance the epoch to the current global epoch.
     * Should only be called when the thread has not been ejected.
     */
    inline void refresh(Alloc& a)
    {
      assert(lock.debug_internal_held());

      auto new_epoch = GlobalEpoch::get();
      auto old_epoch = get_epoch();
      assert(!(old_epoch & EJECTED_BIT));

      if (old_epoch != new_epoch)
      {
        assert(inc_epoch_by(old_epoch, 1) == new_epoch);
        flush_old_epoch(a);
        epoch.store(new_epoch, std::memory_order_release);
      }
    }

    NOINLINE void rejoin_epoch(Alloc& a)
    {
      uint64_t old_epoch = get_epoch();
      assert(old_epoch & EJECTED_BIT);
      assert(lock.debug_internal_held());

      // Clear the ejected bit
      old_epoch = old_epoch & ~EJECTED_BIT;
      Logging::cout() << "Rejoining epoch " << old_epoch << Logging::endl;

      // Read the global epoch
      auto new_epoch = GlobalEpoch::get();

      // Remove the ejected bit, this will prevent other threads from advancing
      // the epoch continually using an up-to-date snapshot of the epoch.
      // Need to prevent subsequent load of global epoch occuring before this
      // store.
      epoch.store(new_epoch, std::memory_order_seq_cst);

      uint64_t guessed_epoch;
      do
      {
        guessed_epoch = new_epoch;
        // Re-read the global epoch
        new_epoch = GlobalEpoch::get();

        // At this point, we know that
        //   new_epoch == GlobalEpoch::get()
        //   || new_epoch + 1 == GlobalEpoch::get()
        //   || (guessed_epoch == new_epoch + 1)   (A)
        // If guessed_epoch != new_epoch+1, then the global epoch can be
        // advanced at most once from this point. For the (A) case, this would
        // require a 2^63 wrap around it in sequence of three instructions.

        // Publish we are in the latest epoch
        epoch.store(new_epoch, std::memory_order_seq_cst);

        // Remove highly unlikely case (A) from above
        // by retrying.
      } while (guessed_epoch == inc_epoch_by(new_epoch, 1));

      // Hence we are now in state 1 or 2 from the comment at the top of the
      // file.

      // Flush all the old epochs, we might have been ejected for a while, so
      // there could be more than one.
      size_t max_steps = 4;
      do
      {
        flush_old_epoch(a);
        old_epoch = inc_epoch_by(old_epoch, 1);
      } while ((old_epoch != new_epoch) && (--max_steps > 0));
    }

    /**
     * Ejects the current thread from the epoch system.
     * Must hold the lock (internal or external).
     */
    void eject()
    {
      epoch.store(get_epoch() | EJECTED_BIT, std::memory_order_release);
    }

    /**
     * Returns true if o is in the epoch e
     * or if o has been ejected.
     * otherwise returns false.
     */
    static bool in_epoch(LocalEpoch* o, uint64_t e)
    {
      assert((e & EJECTED_BIT) == 0);
      auto oe = o->get_epoch();
      return (e == oe) || ((oe & EJECTED_BIT) == EJECTED_BIT);
    }

    /**
     * Returns true if o is in the epoch e
     * or if o has been ejected.
     *
     * It will also attempt to eject o if it is not in the epoch e.
     */
    static bool in_epoch_try_eject(LocalEpoch* o, uint64_t e)
    {
      if (in_epoch(o, e))
      {
        return true;
      }

      if (o->lock.try_external_acquire())
      {
        if (!in_epoch(o, e))
        {
          Logging::cout() << "Ejecting other thread: found" << o->get_epoch()
                          << " requires " << e << Logging::endl;
          o->eject();
        }

        o->lock.external_release();
        return true;
      }

      return false;
    }

    /**
     * Try to advance the global epoch.
     * If try_eject is true, then we will attempt to eject threads that are not
     * in the latest epoch.
     * Returns true if the global epoch was advanced.
     */
    bool try_advance_global_epoch(bool try_eject)
    {
      // Client must have already locked the epoch
      assert(lock.debug_internal_held());

      uint64_t e = get_epoch();

      // Check that all threads are in the same epoch as us
      if (try_eject)
      {
        if (!forall<uint64_t, in_epoch_try_eject>(e))
          return false;
      }
      else
      {
        if (!forall<uint64_t, in_epoch>(e))
          return false;
      }

      // Advance the global epoch
      // Note multiple threads could be attempting this write at the same time.
      // But they will all be attempting to write the same value.
      auto next_epoch = inc_epoch_by(e, 1);
      assert((GlobalEpoch::get() == e) || GlobalEpoch::get() == e + 1);
      GlobalEpoch::set(next_epoch);
      return true;
    }

    void use_epoch_rare(Alloc& a, uint64_t old_epoch)
    {
      if ((old_epoch & EJECTED_BIT) == 0)
      {
        refresh(a);
      }
      else
      {
        rejoin_epoch(a);
      }
    }

    NOINLINE void release_epoch_rare(Alloc& a)
    {
      refresh(a);

      if (advance_is_sensible())
      {
        if (try_advance_global_epoch(advance_is_urgent()))
          refresh(a);
      }
    }

    void debug_check_count()
    {
#ifndef NDEBUG
      {
        size_t sum = 0;

        for (auto i : unusable)
          sum += i;

        auto len = delete_list.length();
        if (sum != len)
          Logging::cout() << "debug_check_cout: Unusable: " << sum
                          << " list.length " << len << Logging::endl;

        assert(sum == len);
      }
      {
        size_t sum = 0;

        for (auto i : to_dec)
          sum += i;

        auto len = dec_list.length();

        if (sum != len)
          Logging::cout() << "debug_check_cout: to_dec: " << sum
                          << " list.length " << len << Logging::endl;

        assert(sum == dec_list.length());
      }
#endif
    }
  };

  using LocalEpochPool = snmalloc::Pool<LocalEpoch, snmalloc::Alloc::Config>;

  template<typename T, bool predicate(LocalEpoch* p, T t)>
  bool LocalEpoch::forall(T t)
  {
    auto curr = LocalEpochPool::iterate();

    while (curr != nullptr)
    {
      if (!predicate(curr, t))
        return false;

      curr = LocalEpochPool::iterate(curr);
    }

    return true;
  }

  /**
   * Handles lifetime management of the LocalEpoch structure.
   */
  class ThreadLocalEpoch
  {
  private:
    friend class Epoch;
    LocalEpoch* ptr;

    ThreadLocalEpoch()
    {
      ptr = LocalEpochPool::acquire();
    }

    ~ThreadLocalEpoch()
    {
      ptr->eject();
      LocalEpochPool::release(ptr);
    }
  };

  /**
   * RAII wrapper for an epoch.
   */
  class Epoch
  {
  private:
    Alloc& alloc;
    LocalEpoch* local_epoch;

  public:
    Epoch(const Epoch&) = delete;
    Epoch& operator=(const Epoch&) = delete;

    Epoch(Alloc& a) : alloc(a)
    {
      static thread_local ThreadLocalEpoch thread_local_epoch;
      yield();
      local_epoch = thread_local_epoch.ptr;
      local_epoch->use_epoch(a);
    }

    ~Epoch()
    {
      if (local_epoch == nullptr)
        return;

      yield();
      local_epoch->release_epoch(alloc);
      yield();
    }

    uint64_t get_local_epoch_epoch()
    {
      return local_epoch->epoch;
    }

    void delete_in_epoch(void* object)
    {
      local_epoch->add_to_delete_list(object);
    }

    void dec_in_epoch(Object* object)
    {
      local_epoch->add_to_dec_list(alloc, object);
    }

    /**
     * Empties all the delayed operations. This does not wait until the epoch
     * has been advanced, and should only be called when this is safe due to
     * other synchronization, such as during teardown.
     */
    static void flush(Alloc& a)
    {
      auto curr = LocalEpochPool::iterate();

      while (curr != nullptr)
      {
        // There are four epoch that can be cleared.
        for (int i = 0; i < 4; i++)
          curr->flush_old_epoch(a);

        curr->eject();

        curr = LocalEpochPool::iterate(curr);
      }
    }
  };
} // namespace verona::rt
