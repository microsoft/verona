// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "../ds/asymlock.h"
#include "../ds/queue.h"
#include "../test/systematic.h"
#include "region/immutable.h"

#include <snmalloc.h>

namespace verona::rt
{
  static constexpr uint64_t EJECTED_BIT = 0x8000000000000000;

  static uint64_t inc_epoch_by(uint64_t epoch, uint64_t i)
  {
    // Maybe overflowing after the addition; need ~EJECTED_BIT for modulo
    return (epoch + i) & ~EJECTED_BIT;
  }

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

    static void advance()
    {
      auto global_e = get();
      global_e = inc_epoch_by(global_e, 3);
      set(global_e);
    }
  };

  class LocalEpoch : public Pooled<LocalEpoch>
  {
  private:
    struct InnerNode
    {
      InnerNode* next;
    };

    struct DecNode
    {
      DecNode* next;
      Object* o;
    };

    friend class ThreadLocalEpoch;
    friend class Epoch;

    Queue<InnerNode> delete_list;
    Queue<InnerNode> dec_list;
    // Providing heuristic for advancing the epoch. Currently, we only look at
    // one slot to determine if we should advance the epoch (see
    // advance_is_sensible()), but we keep the history here so that better
    // heuristic could be applied later.
    size_t pressure[4] = {0, 0, 0, 0};
    size_t unusable[4] = {0, 0, 0, 0};
    size_t to_dec[4] = {0, 0, 0, 0};
    uint8_t index = 0;

    std::atomic<uint64_t> epoch = EJECTED_BIT;
    AsymmetricLock lock;

    template<typename T, bool predicate(LocalEpoch* p, T t)>
    static bool forall(T t);

    void add_to_delete_list(void* p)
    {
      delete_list.enqueue((InnerNode*)p);
      (*get_unusable(2))++;
      (*get_pressure(2))++;
      debug_check_count();
    }

    void add_to_dec_list(Alloc* alloc, Object* p)
    {
      auto node = (DecNode*)alloc->alloc<sizeof(DecNode)>();
      node->o = p;
      dec_list.enqueue((InnerNode*)node);
      (*get_to_dec(2))++;
      debug_check_count();
    }

    inline void use_epoch(Alloc* a)
    {
      lock.internal_acquire();

      auto new_epoch = GlobalEpoch::get();
      auto old_epoch = get_epoch();

      if (old_epoch != new_epoch)
      {
        if (lock.internal_count() == 1)
          use_epoch_rare(a, old_epoch, new_epoch);
      }
    }

    inline void release_epoch(Alloc* a)
    {
      if (advance_is_sensible())
      {
        if (lock.internal_count() == 1)
          release_epoch_rare(a);
      }

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

    void advance_epoch(Alloc* alloc)
    {
      debug_check_count();

      {
        auto cell = get_unusable(0);
        auto usable = *cell;

        for (size_t n = 0; n < usable; n++)
          alloc->dealloc(delete_list.dequeue());

        *cell = 0;

        *get_pressure(0) = 0;
      }

      {
        auto cell = get_to_dec(0);
        auto usable = *cell;

        for (size_t n = 0; n < usable; n++)
        {
          auto dn = (DecNode*)dec_list.dequeue();
          auto o = dn->o;
          alloc->dealloc<sizeof(DecNode)>(dn);
          Systematic::cout() << "Delayed decref on " << o << std::endl;
          Immutable::release(alloc, o);
        }

        *cell = 0;
      }

      index = (index + 1) & 3;
    }

    void add_pressure()
    {
      (*get_pressure(2))++;
    }

    // TODO: Add a proper heuristic here
    bool advance_is_sensible()
    {
#ifdef USE_SYSTEMATIC_TESTING
      return Systematic::coin(4);
#else
      return *get_pressure(2) > 128;
#endif
    }

    bool advance_is_urgent()
    {
#ifdef USE_SYSTEMATIC_TESTING
      return Systematic::coin(7);
#else
      return *get_pressure(2) > 1024;
#endif
    }

    uint64_t get_epoch()
    {
      return epoch.load(std::memory_order_acquire);
    }

    NOINLINE
    void refresh_rare(Alloc* a, uint64_t old_epoch, uint64_t new_epoch)
    {
      advance_epoch(a);

      if (inc_epoch_by(old_epoch, 1) != new_epoch)
      {
        advance_epoch(a);

        if (inc_epoch_by(old_epoch, 1) != new_epoch)
        {
          advance_epoch(a);
        }
      }

      epoch.store(new_epoch, std::memory_order_release);
    }

    inline void refresh(Alloc* a)
    {
      assert(lock.debug_internal_held());

      auto new_epoch = GlobalEpoch::get();
      auto old_epoch = get_epoch();

      if (old_epoch != new_epoch)
      {
        refresh_rare(a, old_epoch, new_epoch);
      }
    }

    NOINLINE void rejoin_epoch(Alloc* a, uint64_t old_epoch)
    {
      epoch.store(old_epoch & ~EJECTED_BIT, std::memory_order_release);

      do
      {
        refresh(a);
        // TODO FENCE?
      } while (epoch.load(std::memory_order_relaxed) != GlobalEpoch::get());
    }

    void eject()
    {
      epoch.store(get_epoch() | EJECTED_BIT, std::memory_order_release);
    }

    static bool not_in_epoch(LocalEpoch* o, uint64_t e)
    {
      // This implicitly checks for EJECTED_BIT being set
      assert((e & EJECTED_BIT) == 0);
      return e != o->get_epoch();
    }

    static bool not_in_epoch_try_eject(LocalEpoch* o, uint64_t e)
    {
      if (not_in_epoch(o, e))
      {
        return true;
      }

      if (o->lock.try_external_acquire())
      {
        if (!not_in_epoch(o, e))
        {
          Systematic::cout() << "Ejecting other thread" << std::endl;
          o->eject();
        }

        o->lock.external_release();
        return true;
      }

      return false;
    }

    void advance_global_epoch(bool try_eject)
    {
      // Client must have already locked the epoch
      assert(lock.debug_internal_held());

      uint64_t e = get_epoch();
      uint64_t e_prev = (e - 1) & ~EJECTED_BIT;

      if (try_eject)
      {
        if (!forall<uint64_t, not_in_epoch_try_eject>(e_prev))
          return;
      }
      else
      {
        if (!forall<uint64_t, not_in_epoch>(e_prev))
          return;
      }

      auto next_epoch = inc_epoch_by(e, 1);
      assert((GlobalEpoch::get() == e) || GlobalEpoch::get() == e + 1);
      GlobalEpoch::set(next_epoch);
    }

    void use_epoch_rare(Alloc* a, uint64_t old_epoch, uint64_t new_epoch)
    {
      if ((old_epoch & EJECTED_BIT) == 0)
      {
        refresh_rare(a, old_epoch, new_epoch);
      }
      else
      {
        rejoin_epoch(a, old_epoch);
      }
    }

    NOINLINE void release_epoch_rare(Alloc* a)
    {
      refresh(a);

      if (advance_is_sensible())
      {
        advance_global_epoch(advance_is_urgent());
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

        assert(sum == delete_list.length());
      }
      {
        size_t sum = 0;

        for (auto i : to_dec)
          sum += i;

        assert(sum == dec_list.length());
      }
#endif
    }
  };

  static inline Pool<LocalEpoch>& global_epoch_set()
  {
    return *Singleton<Pool<LocalEpoch>*, Pool<LocalEpoch>::make>::get();
  }

  template<typename T, bool predicate(LocalEpoch* p, T t)>
  bool LocalEpoch::forall(T t)
  {
    auto curr = global_epoch_set().iterate();

    while (curr != nullptr)
    {
      if (!predicate(curr, t))
        return false;

      curr = global_epoch_set().iterate(curr);
    }

    return true;
  }

  class ThreadLocalEpoch
  {
  private:
    friend class Epoch;
    LocalEpoch* ptr;

    ThreadLocalEpoch()
    {
      ptr = global_epoch_set().acquire();
    }

    ~ThreadLocalEpoch()
    {
      ptr->eject();
      global_epoch_set().release(ptr);
    }
  };

  class Epoch
  {
  private:
    Alloc* alloc;
    LocalEpoch* local_epoch;

  public:
    Epoch(Alloc* a) : alloc(a)
    {
      static thread_local ThreadLocalEpoch thread_local_epoch;
      yield();
      local_epoch = thread_local_epoch.ptr;
      local_epoch->use_epoch(a);
    }

    ~Epoch()
    {
      yield();
      local_epoch->release_epoch(alloc);
      yield();
    }

    void add_pressure()
    {
      local_epoch->add_pressure();
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

    void flush_local()
    {
      for (int i = 0; i < 4; i++)
        local_epoch->advance_epoch(alloc);
    }

    static void flush(Alloc* a)
    {
      // This should only be called when no threads are using the epoch, for
      // example when cleaning up before process termination.
      auto curr = global_epoch_set().iterate();

      while (curr != nullptr)
      {
        for (int i = 0; i < 4; i++)
          curr->advance_epoch(a);

        curr = global_epoch_set().iterate(curr);
      }
    }
  };
} // namespace verona::rt
