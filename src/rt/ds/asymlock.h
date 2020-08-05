// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once
#include "barrier.h"

namespace verona::rt
{
  /**
   * AsymmetricLock allows a single owning thread to use the internal
   * acquire/release API. Other threads must use the external API.
   */
  class AsymmetricLock
  {
  private:
    std::atomic<uint64_t> internal_lock = 0;
    std::atomic<bool> external_lock = false;

    NOINLINE void internal_acquire_rare()
    {
      internal_release();

      while (external_lock.exchange(true, std::memory_order_acq_rel))
        snmalloc::Aal::pause();

      internal_lock.store(1, std::memory_order_relaxed);
      external_release();
    }

  public:
    void external_release()
    {
      external_lock.store(false, std::memory_order_release);
    }

    bool try_external_acquire()
    {
      if (internal_lock.load(std::memory_order_acquire) != 0)
        return false;

      if (!external_lock.exchange(true, std::memory_order_acq_rel))
      {
        Barrier::memory();

        if (internal_lock.load(std::memory_order_acquire) == 0)
          return true;

        external_release();
      }

      return false;
    }

    void external_acquire()
    {
      while (external_lock.exchange(true, std::memory_order_acq_rel))
        snmalloc::Aal::pause();

      Barrier::memory();

      while (internal_lock.load(std::memory_order_acquire) != 0)
        snmalloc::Aal::pause();
    }

    void internal_release()
    {
      uint64_t count = internal_lock.load(std::memory_order_relaxed);
      internal_lock.store(count - 1, std::memory_order_release);
    }

    void internal_acquire()
    {
      uint64_t count = internal_lock.load(std::memory_order_relaxed);
      internal_lock.store(count + 1, std::memory_order_relaxed);

      Barrier::compiler();

      if (external_lock.load(std::memory_order_relaxed))
      {
        // Already hold the lock in the reentrant case
        if (count > 0)
          return;

        internal_acquire_rare();
      }
    }

    uint64_t internal_count()
    {
      assert(debug_internal_held());
      return internal_lock.load(std::memory_order_relaxed);
    }

    bool debug_internal_held()
    {
      return internal_lock.load(std::memory_order_relaxed) != 0;
    }
  };
} // namespace verona::rt
