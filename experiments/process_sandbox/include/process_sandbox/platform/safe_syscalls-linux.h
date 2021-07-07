// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#ifdef __linux__

#  include <fcntl.h>
#  include <sys/stat.h>
#  include <sys/syscall.h>
#  include <unistd.h>

/**
 * This file contains per-platform definitions of capability-secure filesystem
 * operations.  Each of these is a wrapper around OS-specific `*at()` system
 * calls that enforce the invariant that the file path must be below the file
 * descriptor that the .
 */
namespace sandbox::platform
{
  class SafeSyscallsLinux
  {
    /**
     * The system call number for openat2
     */
    static int constexpr openat2_number =
#  ifdef __NR_openat2
      __NR_openat2
#  elif defined(__x86_64__)
      // This system call was introduced with Linux 5.6, but Ubuntu 20.04
      // doesn't provide headers that expose it, so hard-code the number
      // for x86-64 until they catch up.
      437
#  else
#    error openat2 system call number not found
#  endif
      ;

    static int constexpr faccessat2_number =
#  ifdef __NR_faccessat2
      __NR_faccessat2
#  elif defined(__x86_64__)
      // This system call was introduced with Linux 5.6, but Ubuntu 20.04
      // doesn't provide headers that expose it, so hard-code the number
      // for x86-64 until they catch up.
      439
#  else
#    error faccessat2 system call number not found
#  endif
      ;

    /**
     * The flag for RESOLVE_BENEATH.  We should get this from the Linux
     * headers, but it's part of the kernel ABI and so it's safe to specify
     * here and Ubuntu 20.04 doesn't include the correct headers.
     */
    static constexpr uint64_t resolve_beneath = 0x08;

    /**
     * The `O_PATH` constant, which is not always exposed by Linux headers.
     * This allows `open` / `openat` to open a file descriptor for use with
     * `*at` calls.
     */
    static constexpr int o_path = 010000000;

    /**
     * The `AT_EMPTY_PATH` flag.  This allows `*at` calls to take an empty
     * path argument and operate on the file descriptor.
     */
    static constexpr int at_empty_path = 0x1000;

    /**
     * The structure for the `how` argument to `openat2`.  The kernel may
     * extend this in the future but the system call takes a size argument
     * and so it's safe for us to use this version even if it becomes stale
     * in the future.
     *
     * As with the other `openat2`-related things, we define our own
     * because Ubuntu 20.04 does not install up-to-date headers.
     */
    struct openat_how
    {
      uint64_t flags;
      uint64_t mode;
      uint64_t resolve;
    };

  public:
    /**
     * Open a file that must be in the filesystem tree identified by `fd`.  The
     * arguments correspond to those of the `openat` call in POSIX.
     */
    static int openat_beneath(
      platform::handle_t fd, const char* path, int flags, mode_t mode)
    {
      openat_how how{static_cast<uint64_t>(flags), mode, resolve_beneath};
      return syscall(openat2_number, fd, path, &how, sizeof(how));
    }

    /**
     * Check for access of a file.  The arguments correspond to those of the
     * `faccessat` call in POSIX.
     */
    static int faccessat_beneath(
      platform::handle_t fd, const char* path, int mode, int flags = 0)
    {
      int eno;
      int ret;
      {
        Handle path_handle{openat_beneath(fd, path, o_path, 0)};
        if (!path_handle.is_valid())
        {
          eno = errno;
          ret = -1;
        }
        else
        {
          ret = syscall(
            faccessat2_number, path_handle.fd, "", mode, flags | at_empty_path);
          eno = errno;
        }
      }
      // Set errno to the value from the call that we care about, not any value
      // set by `close` in `Handle`'s destructor.
      errno = eno;
      return ret;
    };

    /**
     * Check the properties of a file.  The arguments correspond to those of
     * the `fstatat_beneath` call in POSIX.
     */
    static int fstatat_beneath(
      platform::handle_t fd, const char* path, struct stat* sb, int flags)
    {
      int eno;
      int ret;
      {
        Handle path_handle{openat_beneath(fd, path, o_path, 0)};
        if (!path_handle.is_valid())
        {
          eno = errno;
          ret = -1;
        }
        else
        {
          ret = fstatat(path_handle.fd, "", sb, flags | at_empty_path);
          eno = errno;
        }
      }
      // Set errno to the value from the call that we care about, not any
      // value set by `close` in `Handle`'s destructor.
      errno = eno;
      return ret;
    }
  };
}

#endif
