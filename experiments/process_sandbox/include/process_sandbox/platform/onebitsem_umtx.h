// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once
#if __has_include(<sys/umtx.h>)
#  include "../helpers.h"

#  include <atomic>
#  include <sys/types.h>
#  include <sys/umtx.h>

namespace sandbox::platform
{
  class UMtxOneBitSem
  {
    /**
     * A userspace semaphore structure.  This type must have the same
     * layout as the _usem2 structure defined by the kernel.  This version
     * is provided so that we can use C++11 atomics.
     */
    struct usem2
    {
      /**
       * The value of the semaphore.  This is incremented on wakes and
       * decremented on waits.  A high bit is used to indicate whether
       * there are waiters.
       */
      std::atomic<uint32_t> count = {0};
      /**
       * Flags.
       */
      uint32_t flags = {USYNC_PROCESS_SHARED};
    } sem;
    static_assert(
      sizeof(usem2) == sizeof(::_usem2),
      "The usem2 structure has the wrong type");

  public:
    UMtxOneBitSem() = default;
    UMtxOneBitSem(bool init)
    {
      sem.count = init;
    }
    void wake()
    {
      uint32_t old = sem.count.fetch_add(1, std::memory_order_release);
      SANDBOX_INVARIANT(
        USEM_COUNT(old) == 0,
        "Waking up one-bit semaphore that's already awake.  Count: {}.",
        USEM_COUNT(old));
      if (old & USEM_HAS_WAITERS)
      {
        int ret = _umtx_op(&sem, UMTX_OP_SEM2_WAKE, 0, NULL, NULL);
        SANDBOX_INVARIANT(ret == 0, "_umtx_op failed: {}", ret);
      }
    }
    bool wait(int milliseconds)
    {
      auto try_lock = [&]() {
        uint32_t count = sem.count.load();
        if (USEM_COUNT(count) > 0)
        {
          assert(USEM_COUNT(count) == 1);
          if (sem.count.compare_exchange_strong(
                count,
                count - 1,
                std::memory_order_acquire,
                std::memory_order_acquire))
          {
            return true;
          }
        }
        return false;
      };
      if (try_lock())
      {
        return true;
      }
      struct
      {
        struct _umtx_time timeout;
        struct timespec remainder;
      } timeout;
      timeout.timeout._clockid = CLOCK_MONOTONIC;
      timeout.timeout._flags = 0;
      timeout.timeout._timeout = {milliseconds / 1000,
                                  (milliseconds % 1000) * 1000000};
      int ret;
      do
      {
        ret = _umtx_op(
          &sem,
          UMTX_OP_SEM2_WAIT,
          0,
          reinterpret_cast<void*>(static_cast<uintptr_t>(sizeof(timeout))),
          &timeout);
        timeout.timeout._timeout = timeout.remainder;
      } while ((ret == -1) && (errno == EINTR));
      assert((ret != -1) || (errno == ETIMEDOUT));
      return try_lock();
    }
  };
}

#endif
