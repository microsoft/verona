// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

/**
 * Platform interface.  This is the top-level include for anything
 * platform-specific.
 */

#include <snmalloc/pal/pal.h>
#ifdef __unix__
#  include <fcntl.h>
#endif
#include <assert.h>
#include <functional>
#ifdef __linux__
#  include <sys/personality.h>
#endif

#pragma once
namespace sandbox
{
  namespace platform
  {
#ifdef __unix__
    /**
     * The `handle_t` type represents a handle used to access OS resources.  On
     * POSIX systems, this is a file descriptor or, more accurately, an integer
     * index into the process's file-descriptor table.
     */
    using handle_t = int;
    /**
     * Class encapsulating a file descriptor.  This handles deallocation.
     */
    struct Handle
    {
      /**
       * The file descriptor that this wraps.  POSIX file descriptors are
       * indexes into a file descriptor table. Negative indexes are invalid and
       * so -1 is used as a default invalid value.
       *
       * This field is specific to the POSIX implementation and so should be
       * used only in POSIX-specific code paths.
       */
      handle_t fd = -1;

      /**
       * Check if this is a valid file descriptor.  This should be used only to
       * check whether this class has been initialised with a valid descriptor:
       * even if the file descriptor is valid at the call, another thread could
       * manipulate the file descriptor table and invalidate it immediately
       * after this function returns.
       *
       * In debug builds, this will check if the file descriptor refers to a
       * valid entry in the file descriptor table, though the above caveats
       * still apply.
       */
      bool is_valid()
      {
        assert(
          ((fd < 0) || (fcntl(fd, F_GETFD) >= 0)) &&
          "File descriptor is a valid index but does not refer to a valid file "
          "descriptor");
        return fd >= 0;
      }
      void reset(handle_t new_fd)
      {
        if (is_valid())
        {
          close(fd);
        }
        fd = new_fd;
      }

      Handle() = default;
      /**
       * Construct a `Handle` from the raw OS handle.  This is explicit to
       * avoid accidentally taking ownership of a handle and closing it when a
       * temporary `Handle` is destroyed.
       */
      explicit Handle(handle_t new_fd) : fd(new_fd) {}

      /**
       * Copy constructor is deleted.  File descriptors are not reference
       * counted and so must have a single deleter.  If a file descriptor needs
       * to be multiply owned, this should be done via a
       * `std::shared_ptr<Handle>`.
       */
      Handle(const Handle&) = delete;

      /**
       * Move constructor.  Takes ownership of a file descriptor.
       */
      Handle(Handle&& other) : fd(other.fd)
      {
        other.fd = -1;
      }

      Handle& operator=(Handle&& other)
      {
        reset(other.fd);
        other.fd = -1;
        return *this;
      }

      Handle& operator=(handle_t new_fd)
      {
        reset(new_fd);
        return *this;
      }

      bool operator==(const handle_t other_fd) const
      {
        return fd == other_fd;
      }

      bool operator==(const Handle& other) const
      {
        return fd == other.fd;
      }

      /**
       * Extract the raw OS handle.  The caller is responsible for any cleanup.
       * This can be done by constructing a new `Handle` from the result of this
       * function.
       */
      handle_t take()
      {
        handle_t ret = fd;
        fd = -1;
        return ret;
      }

      /**
       * Destructor, closes the file descriptor if it is valid.
       */
      ~Handle()
      {
        reset(-1);
      }
    };
#else
#  error Handle type not defined for your platform
#endif
    inline void disable_aslr()
    {
#ifdef __linux__
      int p = personality(0xffffffff);
      personality(p | ADDR_NO_RANDOMIZE);
#endif
    }
  }
}

namespace std
{
  template<>
  struct hash<sandbox::platform::Handle>
  {
    size_t operator()(const sandbox::platform::Handle& x) const
    {
      return std::hash<sandbox::platform::handle_t>()(x.fd);
    }
  };
}

#include "child_process.h"
#include "onebitsem.h"
#include "poller.h"
#include "safe_syscalls.h"
#include "sandbox.h"
#include "shm.h"
#include "socketpair.h"
#include "syscall_context.h"
