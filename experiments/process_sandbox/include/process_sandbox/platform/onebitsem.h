// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
/**
 * This file defines the target platform's implementation of the 1-bit semaphore
 * interface and exports the preferred version as the `OneBitSem` type.
 *
 * Implementations of this type must be possible to store in shared memory
 * between processes.  It is used to implement a simple token-passing model for
 * RPC, where the child sleeps one one semaphore, the parent wakes the child's
 * semaphore and sleeps on another.
 *
 * Implementations of this interface should expose two methods:
 *
 * ```
 * void wake();
 * ```
 * This method wakes up the other side.
 *
 * ```
 * bool wait(int milliseconds);
 * ```
 *
 * Blocks for either the specified number of milliseconds have elapsed or
 * until `wake` is called.  Returns true if the return is in response to a wake
 * event, false if it is in response to a timeout.
 */

#include "onebitsem_futex.h"
#include "onebitsem_umtx.h"
//#include "onebitsem_posix.h"

namespace sandbox
{
  namespace platform
  {
    using OneBitSem =
#ifdef __FreeBSD__
      UMtxOneBitSem
#elif defined(__linux__)
      FutexOneBitSem
#elif defined(__unix__)
      PosixOneBitSem
#else
#  error No one-bit semaphore defined for your platform
#endif
      ;
  }
}
