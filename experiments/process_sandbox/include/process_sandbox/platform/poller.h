// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
/**
 * This file defines the target platform's implementation of the Poller
 * interface and exports the preferred version as the `Poller` type.
 *
 * Implementations of this interface should expose two methods:
 *
 * ```
 * void add(handle_t fd);
 * ```
 * This thread-safe method registers a handle with the poller so that the
 * poller can detect when it either becomes readable or reaches an end-of-file
 * condition.
 *
 * Note that there is currently no mechanism for removing a handle, handles are
 * implicitly removed once they reach their end-of-file (closed by the remote
 * end) state.
 *
 * ```
 * bool poll(handle_t& fd, bool& eof);
 * ```
 *
 * This method may be called by only a single thread at any time and blocks
 * until one of the registered handles is ready or an error occurs.
 */

#include "poller_epoll.h"
#include "poller_kqueue.h"

namespace sandbox
{
  namespace platform
  {
    using Poller =
#ifdef USE_KQUEUE
      KQueuePoller
#elif defined(__linux__)
      EPollPoller
#else
#  error No poller defined for your platform
#endif
      ;
  }
}
