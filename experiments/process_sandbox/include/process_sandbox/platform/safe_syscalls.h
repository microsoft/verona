// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

/**
 * This file contains per-platform definitions of capability-secure filesystem
 * operations.  Each of these is a wrapper around OS-specific `*at()` system
 * calls that enforce the invariant that the file path must be below the file
 * descriptor that the .
 */

#include "safe_syscalls-freebsd.h"
#include "safe_syscalls-linux.h"

namespace sandbox::platform
{
  using SafeSyscalls =
#ifdef __linux__
    SafeSyscallsLinux
#elif defined(__FreeBSD__)
    SafeSyscallsFreeBSD
#else
#  error Safe system calls not implemented for your platform
#endif
    ;
}
