// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#ifdef __FreeBSD__

#  include <fcntl.h>
#  include <sys/stat.h>
#  include <unistd.h>

namespace sandbox::platform
{
  /**
   * FreeBSD implementations of the safe system calls.  These forward
   * trivially to system calls.
   */
  struct SafeSyscallsFreeBSD
  {
    /**
     * Open a file that must be in the filesystem tree identified by `fd`.  The
     * arguments correspond to those of the `openat` call in POSIX.
     */
    static int openat_beneath(
      platform::handle_t fd, const char* path, int flags, mode_t mode)
    {
      return openat(fd, path, flags | O_RESOLVE_BENEATH, mode);
    }

    /**
     * Check for access of a file.  The arguments correspond to those of the
     * `faccessat` call in POSIX.
     */
    static int faccessat_beneath(
      platform::handle_t fd, const char* path, int mode, int flags = 0)
    {
      return faccessat(fd, path, mode, flags | AT_RESOLVE_BENEATH);
    };

    /**
     * Check the properties of a file.  The arguments correspond to those of
     * the `fstatat_beneath` call in POSIX.
     */
    static int fstatat_beneath(
      platform::handle_t fd, const char* path, struct stat* sb, int flags)
    {
      return fstatat(fd, path, sb, flags | AT_RESOLVE_BENEATH);
    }
  };
}

#endif
