// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

/**
 * This file defines the SocketPair interface, which creates a pair of handles
 * to some local bidirectional IPC mechanism.  This is a trivial interface that
 * exposes a single `static` method:
 *
 * static std::pair<Handle, Handle> create();
 */

#include "socketpair_posix.h"

namespace sandbox
{
  namespace platform
  {
    using SocketPair =
#ifdef __unix__
      SocketPairPosix
#else
#  error SocketPair not implemented for your platform
#endif
      ;
  }
}
