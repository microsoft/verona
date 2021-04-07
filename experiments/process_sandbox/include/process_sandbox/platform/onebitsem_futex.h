// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once
#if __has_include(<linux/futex.h>)
#  include "../helpers.h"

#  include <atomic>
#  include <linux/futex.h>
#  include <sys/syscall.h>
#  include <sys/time.h>

namespace sandbox::platform
{
  /**
   * One-bit semaphore implementation using futexes.  This is a simplified
   * version of a futex-based mutex that (intentionally) does not enforce the
   * requirement that the unlock happens in the same thread as the lock.
   */
  class FutexOneBitSem
  {
    /**
     * The value of the semaphore.  This is set to 0 on a wait, 1 on a wake.
     * Attempting to wait while the value is 0 will block.
     */
    std::atomic<int> flag = {0};
    int
    futex_op(int futex_op, int val, const struct timespec* timeout = nullptr)
    {
      return syscall(
        SYS_futex, reinterpret_cast<int*>(&flag), futex_op, val, timeout, 0, 0);
    }

  public:
    FutexOneBitSem() = default;
    FutexOneBitSem(bool init)
    {
      flag = init;
    }
    void wake()
    {
      uint32_t old = flag.fetch_add(1, std::memory_order_release);
      SANDBOX_INVARIANT(
        old == 0,
        "Waking up one-bit semaphore that's already awake.  Count: {}.",
        old);
      futex_op(FUTEX_WAKE, 1);
    }
    bool wait(int milliseconds)
    {
      auto try_lock = [&]() {
        int f = flag.load();
        if (f > 0)
        {
          assert(f == 1);
          return flag.compare_exchange_strong(
            f, f - 1, std::memory_order_acquire, std::memory_order_acquire);
        }
        return false;
      };
      if (try_lock())
      {
        return true;
      }
      // Note: we always retry with the same timeout because futex doesn't give
      // a mechanism for figuring out how much time elapsed if we were
      // interrupted.
      struct timespec timeout = {milliseconds / 1000,
                                 (milliseconds % 1000) * 1000000};
      int ret;
      do
      {
        ret = futex_op(FUTEX_WAIT, 0, &timeout);
      } while ((ret == -1) && (errno == EINTR));
      assert((ret != -1) || ((errno == ETIMEDOUT) || (errno == EAGAIN)));
      return try_lock();
    }
  };
}

#endif
